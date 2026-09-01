#!/usr/bin/env bash
# find_robot_ip.sh — Discover the ROSPug's IP on the Ethernet interface.
#
# Run this on the HOST (not inside Docker) with the robot powered on.
# Usage:  bash find_robot_ip.sh [ethernet_iface]
# Default interface: enp2s0

set -euo pipefail

IFACE="${1:-enp2s0}"

# Subnets to scan — covers Hiwonder AP-mode default (149.x),
# and both static IPs currently on enp2s0 (123.x and 1.x).
SUBNETS=(
    "192.168.149.0/24"
    "192.168.123.0/24"
    "192.168.1.0/24"
)

# ── Helper: add a temporary IP to reach a subnet if not already present ──
add_temp_ip() {
    local subnet="$1"            # e.g. 192.168.149.0/24
    local probe_ip="$2"          # e.g. 192.168.149.200
    if ! ip addr show "$IFACE" | grep -q "${probe_ip%/*}"; then
        echo "  → Adding temporary $probe_ip to $IFACE to reach $subnet ..."
        sudo ip addr add "$probe_ip" dev "$IFACE" 2>/dev/null || true
        echo "$probe_ip"         # caller can clean up later
    fi
}

echo "============================================================"
echo "  ROSPug Network Discovery"
echo "  Interface : $IFACE"
echo "  $(ip addr show "$IFACE" 2>/dev/null | grep 'inet ' | awk '{print "Current IPs:", $2}' | tr '\n' ' ')"
echo "============================================================"

ADDED_IP=""

# ── Step 1: mDNS (avahi) — robot may announce hiwonder.local ─────────────
echo ""
echo "[1/4] Checking mDNS for common Hiwonder hostnames ..."
for hostname in hiwonder.local raspberrypi.local rospug.local pug.local; do
    ip=$(avahi-resolve -n "$hostname" 2>/dev/null | awk '{print $2}')
    if [[ -n "$ip" ]]; then
        echo "  ✓ Found via mDNS: $hostname → $ip"
        echo ""
        echo "ROBOT_IP=$ip"
        echo "HOST_ETH_IP=$(ip addr show "$IFACE" | grep 'inet ' | head -1 | awk '{print $2}' | cut -d/ -f1)"
        exit 0
    fi
done
echo "  (no mDNS response)"

# ── Step 2: Current ARP neighbour table ──────────────────────────────────
echo ""
echo "[2/4] Checking ARP neighbour table for entries on $IFACE ..."
arp_hits=$(ip neighbor show dev "$IFACE" 2>/dev/null | grep -v "FAILED\|INCOMPLETE" || true)
if [[ -n "$arp_hits" ]]; then
    echo "$arp_hits" | while read -r line; do
        ip_found=$(echo "$line" | awk '{print $1}')
        echo "  ✓ ARP neighbour: $ip_found"
    done
else
    echo "  (no live ARP entries on $IFACE)"
fi

# ── Step 3: Ping sweep each subnet ───────────────────────────────────────
echo ""
echo "[3/4] Ping-sweeping subnets (this takes ~10 seconds) ..."

FOUND_IPS=()

# Ensure we have an IP in each subnet before scanning
temp_ips_added=()
for subnet in "${SUBNETS[@]}"; do
    base="${subnet%.*}"          # e.g. 192.168.149
    case "$base" in
        "192.168.149") probe="192.168.149.200/24" ;;
        "192.168.123") probe="" ;;  # already have 192.168.123.100
        "192.168.1")   probe="" ;;  # already have 192.168.1.100
        *) probe="" ;;
    esac
    if [[ -n "$probe" ]]; then
        added=$(add_temp_ip "$subnet" "$probe" || true)
        [[ -n "$added" ]] && temp_ips_added+=("$added")
    fi

    echo "  Scanning $subnet ..."
    while IFS= read -r line; do
        found_ip=$(echo "$line" | grep "report for" | awk '{print $5}')
        [[ -n "$found_ip" ]] && FOUND_IPS+=("$found_ip") && echo "  ✓ Host found: $found_ip"
    done < <(nmap -sn "$subnet" 2>/dev/null || true)
done

# Clean up temporary IPs
for temp_ip in "${temp_ips_added[@]}"; do
    sudo ip addr del "$temp_ip" dev "$IFACE" 2>/dev/null || true
done

# ── Step 4: SSH probe known Hiwonder default IPs ────────────────────────
echo ""
echo "[4/4] SSH-probing Hiwonder common default IPs ..."
DEFAULT_IPS=("192.168.149.1" "192.168.0.100" "192.168.123.1")
for candidate in "${DEFAULT_IPS[@]}"; do
    if ssh -o ConnectTimeout=2 -o StrictHostKeyChecking=no \
           -o BatchMode=yes "hiwonder@$candidate" "echo ok" 2>/dev/null; then
        echo "  ✓ SSH responded: hiwonder@$candidate"
        FOUND_IPS+=("$candidate")
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
if [[ ${#FOUND_IPS[@]} -eq 0 ]]; then
    echo "  ✗  No live hosts found on any subnet."
    echo ""
    echo "  Troubleshooting:"
    echo "  1. Confirm the ROSPug is powered on (blue LED on controller board)."
    echo "  2. Check the Ethernet cable is firmly seated at both ends."
    echo "  3. If the robot uses a different subnet, add it to SUBNETS in this script."
    echo "  4. Try: sudo arp-scan --interface=$IFACE --localnet"
else
    echo "  Found ${#FOUND_IPS[@]} candidate(s):"
    for ip in "${FOUND_IPS[@]}"; do
        echo "    $ip"
    done
    echo ""
    ROBOT_IP="${FOUND_IPS[0]}"
    HOST_ETH_IP=$(ip addr show "$IFACE" | grep 'inet ' | head -1 | awk '{print $2}' | cut -d/ -f1)
    echo "  → Recommended export commands:"
    echo ""
    echo "    export ROBOT_IP=$ROBOT_IP"
    echo "    export HOST_ETH_IP=$HOST_ETH_IP"
    echo ""
    echo "  → Launch real-robot container:"
    echo ""
    echo "    ROBOT_IP=$ROBOT_IP HOST_ETH_IP=$HOST_ETH_IP \\"
    echo "      docker compose -f docker-compose.real.yml run --rm rospug bash"
fi
echo "============================================================"
