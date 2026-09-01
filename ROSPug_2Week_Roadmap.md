# ROSPug Domain Randomization Project — 2-Week Roadmap (with GitHub Copilot)

## Honest scope note (read this first)

The originally discussed project (real-hardware sim-to-real transfer + a "reality gap estimator" tuned from real trials) is a multi-month undertaking for someone starting from zero. To actually finish something real in 2 weeks, this roadmap targets a scoped, defensible version:

**Core deliverable (must finish):** Train three RL walking policies for ROSPug in Gazebo — (A) fixed physics, (B) fixed-range domain randomisation, (C) Sensitivity-Weighted ADR — then prove C is more robust than B, and B more robust than A, on *held-out* conditions none of them trained on exactly. The novel contribution is SW-ADR: using per-parameter return variance as a brittleness signal to direct differential curriculum expansion, which has not been published on any quadruped platform.

**Stretch goal (only if days 1–12 go well):** Deploy Policy B and C to the real ROSPug and compare walking stability on untrained surfaces against the original fixed-gait controller.

**Cut entirely for this timeline:** the "reality gap estimator" that uses real trials to auto-tune the randomization ranges. That's a natural "future work" section, not something to build now.

If you get through the core deliverable early, treat the stretch goal as bonus — don't let it block finishing the write-up.

---

## Daily structure

Each day below has: **Goal** → **Copilot prompts** (paste these into Copilot Chat, adjusting file paths as needed) → **Testing/exit criteria** (don't move on until this passes) → **What breaks most often**.

Work in a single git repo from day 1 so Copilot has your actual code as context — it gives much better suggestions once it can see your existing files than in a blank chat.

---

## Step1 : Day 0 (half day) — Environment setup

**Goal:** ROS1 + Gazebo + Python RL stack installed and talking to each other. No ROSPug-specific code yet.

**Steps:**
1. Install ROS1 Noetic (Ubuntu 20.04) or use a Docker image if your machine can't run Noetic natively — ROSPug's stack is ROS1.
2. Clone ROSPug's repo and its Gazebo simulation packages:
   - https://github.com/Hiwonder-docs/ROSPug
3. Install Python RL tooling: `pip install stable-baselines3 gymnasium`.

**Copilot prompts:**
- *"I'm setting up ROS1 Noetic with Gazebo on Ubuntu 20.04 for a robot simulation project. Write a bash script that checks whether ROS1 and Gazebo are installed, and if not, installs `ros-noetic-desktop-full` and initializes rosdep."*
- *"Write a minimal Dockerfile based on `osrf/ros:noetic-desktop-full` that also installs `stable-baselines3`, `gymnasium`, and `catkin-tools`, for a ROS1 + Gazebo + RL development environment."* (use this if native install is painful)

**Testing/exit criteria:** `roscore` starts, `gazebo` opens an empty world, `python -c "import stable_baselines3"` runs with no error.

**What breaks most often:** ROS1 Noetic only officially supports Ubuntu 20.04 — if you're on a newer OS, the Docker route above will save you a full day of dependency hell. Don't fight this on day 0; just use Docker.

---

##step2 : Day 1–2 — Get ROSPug walking in Gazebo with its existing controller

**Goal:** ROSPug's simulated model loads in Gazebo and walks using its *original, non-RL* gait controller. This is your sanity-check baseline — if this doesn't work, nothing downstream will.

**Copilot prompts:**
- *"Here is ROSPug's package structure [paste `catkin_ws/src/rospug/` folder listing]. Help me write the `roslaunch` command to spawn ROSPug's URDF/SDF model into an empty Gazebo world."*
- *"I'm getting this Gazebo spawn error: [paste exact error]. My URDF file is [paste rospug.urdf.xacro]. What's wrong?"*
- *"Write a simple ROS1 Python node that publishes to ROSPug's velocity command topic [paste topic name from `rostopic list`] to make it walk forward for 3 seconds, then stop."*

**Testing strategy:**
- Run `roslaunch rospug_gazebo rospug_world.launch` (or equivalent — Copilot will help find the right launch file name from the repo structure).
- Confirm in `rostopic list` that joint states, IMU, and cmd_vel topics exist.
- Run your test node from the third prompt above; ROSPug should visibly walk forward in Gazebo.

**Exit criteria:** You can command ROSPug to walk forward/turn in Gazebo using the existing controller, and you can read back joint angles and IMU orientation live via `rostopic echo`.

**What breaks most often:** Xacro/URDF path errors (missing mesh files) — Copilot is very good at fixing these once you paste the exact error text, don't paraphrase it.

---

##step3 : Day 3–4 — Build a Gym-style RL environment wrapper around ROSPug/Gazebo

**Goal:** A Python class that lets Stable-Baselines3 treat ROSPug-in-Gazebo like any other RL environment: `reset()`, `step(action)`, returns `(observation, reward, done, info)`.

**Copilot prompts:**
- *"I have a ROS1 robot (quadruped) in Gazebo that I can control via cmd_vel and read joint states + IMU from via rostopic. Help me write a Python class inheriting from `gymnasium.Env` that wraps this as an RL environment: `reset()` should reset the Gazebo world and robot pose using the `/gazebo/reset_world` service, `step(action)` should publish a velocity/joint command and wait one control cycle, then return the new observation."*
- *"Design a reward function for a quadruped walking task: reward forward velocity, penalize falling (body roll/pitch beyond a threshold ends the episode with a penalty), and lightly penalize energy use (sum of squared joint velocities). Write this as a `_compute_reward()` method."*
- *"Write a quick test script that runs my new `RosPugEnv` for 5 random-action episodes and prints the reward each step, to confirm the environment doesn't crash before I start training."*

**Testing strategy:**
- Run the random-action test script. It doesn't need to walk well (random actions will look like a flailing robot falling over) — you're only testing that `reset()`/`step()` don't crash and that reward/done signals look sane (falling should terminate the episode and return a negative reward).
- Time how long one episode takes in wall-clock time — this tells you how many training steps are realistically possible in the time you have (this matters a lot for scoping day 5–7).

**Exit criteria:** 20 random-action episodes run back-to-back with no crash, and falling reliably triggers `done=True`.

**What breaks most often:** Gazebo's reset service is slow and sometimes leaves physics in a bad state — Copilot can help you add a small `rospy.sleep()` and a settle-check after reset if episodes start with the robot already tipped over.

---

##Step4 : Day 5–7 — Train baseline PPO policy (no randomization)

**Goal:** A policy that can walk forward a reasonable distance in Gazebo, trained on fixed (non-randomized) physics parameters. This is your **Policy A**, the baseline you'll compare against.

**Copilot prompts:**
- *"Using Stable-Baselines3, write a training script that trains a PPO agent on my custom `RosPugEnv` (already implemented in `rospug_env.py`). Include TensorBoard logging and periodic checkpoint saving every 10,000 steps."*
- *"My PPO training reward is flat / not increasing after 50,000 steps [paste a screenshot description or the reward curve numbers]. Given my reward function [paste it], what are likely causes and what should I try first — reward scaling, learning rate, or episode length?"*
- *"Write an evaluation script that loads a saved PPO checkpoint and runs it for 10 episodes in the Gazebo environment, logging total distance walked and whether the robot fell, for each episode."*

**Testing strategy:**
- Watch training in TensorBoard (`tensorboard --logdir ./logs`) — reward should trend upward, even if noisily.
- Every ~30–60 minutes of training, run the evaluation script and *visually watch* Gazebo — numbers going up on a graph don't guarantee it looks like walking. Trust your eyes over the reward curve early on.
- Budget for reward-shaping iteration — this is normal and expected, not a sign something is broken.

**Exit criteria:** ROSPug reliably walks forward for several body-lengths without falling, in the *same* simulated conditions it was trained on. It does not need to look elegant — "doesn't fall over and makes forward progress" is a real, sufficient milestone.

**What breaks most often:** This is the highest-risk step in the whole project. If by end of day 7 you have a policy that walks even imperfectly, move on — don't keep polishing gait quality, since days 8–10 are what actually make this a "domain randomization" project rather than "I trained a walking robot" (which is a fine fallback if you truly run out of time, but isn't the assignment).

---

##step5: Day 8–9 — Add domain randomization and train Policy B

**Goal:** Modify the environment so that at the start of every episode, physics parameters (friction, body mass, servo response latency) are resampled from a range. Train a second PPO policy on this randomized environment.

**Copilot prompts:**
- *"Modify my `reset()` method in `RosPugEnv` to also randomize Gazebo physics parameters before each episode: ground friction (range 0.4–1.2), robot body mass (±15% of nominal), and simulated servo command delay (0–50ms). Use Gazebo's `/gazebo/set_physics_properties` service and the SDF `<mu1>/<mu2>` friction tags — show me how to set these programmatically between episodes."*
- *"I want to add artificial servo latency to my action pipeline in `step()` — delay applying the commanded joint targets by a random 0–50ms sampled fresh each episode, to simulate real servo response lag. How do I implement this cleanly in a ROS1 control loop without breaking the timing?"*
- *"Train a second PPO agent identical in hyperparameters to my baseline, but using the randomized environment. Save it under a different name/log directory so I can compare both later."*

**Testing strategy:**
- Confirm randomization is actually happening: log the sampled friction/mass/latency values at the start of every episode for the first 10 episodes and print them — verify they're different each time, not stuck at one value (a common bug).
- Expect training to look *messier* and converge *slower* than the baseline — this is expected and is the whole point (the policy is solving a harder, more general problem), not a sign of failure.

**Exit criteria:** Policy B walks forward without falling in *most* of the randomized conditions it was trained across (it doesn't need to be perfect in every single one).

**What breaks most often:** Randomizing too aggressively (e.g., near-zero friction) can make the task unsolvable and training never converges. If B refuses to learn at all after a few hours, ask Copilot to help you narrow the randomization ranges — this is a legitimate, reportable design decision, not cheating.

---

##step6: Day 10–11 — Train Policy C: Sensitivity-Weighted Adaptive Domain Randomisation

**Goal:** Train a third PPO policy (Policy C) using Sensitivity-Weighted ADR (SW-ADR) — a novel extension of standard domain randomisation where **per-parameter brittleness signals** drive differential curriculum expansion. Policy C is the primary novel contribution of the project.

**What makes this novel vs. standard ADR:**
Standard ADR (Akkaya et al. 2019) tracks aggregate success rate and expands or contracts all DR ranges at the same rate. SW-ADR instead tracks, for each physics parameter independently, the variance of episode returns when that parameter is sampled near the boundary of its current range. A high boundary variance means the policy's performance is unstable at the edge of that dimension — it is *brittle* to that parameter. SW-ADR expands brittle dimensions faster and holds stable dimensions, creating a targeted curriculum that pushes the policy hardest where it is weakest.

This has not been published on any quadruped platform, budget or otherwise.

**Mechanism:**
1. Maintain a rolling buffer of `(dr_params, episode_return)` tuples (last 50 episodes)
2. For each DR dimension d, split recent episodes into *boundary episodes* (d sampled in top/bottom 20% of current range) and *interior episodes*
3. `sensitivity_d = Var(returns | boundary) − Var(returns | interior)`
4. Every 50 episodes, normalise sensitivities and expand each dimension proportionally to its sensitivity weight — most brittle dimension expands fastest
5. Log current range bounds and per-dimension sensitivity to TensorBoard every update

**Copilot prompts:**
- *"I have `RosPugEnvDR` which samples 4 physics parameters (mass_frac, latency, cfm, erp) each episode and returns `info['dr_params']` and the episode return. Create a subclass `RosPugEnvADR` that maintains a rolling window of (dr_params, return) tuples. Every 50 episodes, compute per-parameter sensitivity as Var(returns | param in top 20% of range) minus Var(returns | param in interior). Normalise sensitivities to weights summing to 1. Expand each parameter range proportionally to its sensitivity weight, capped at predefined maximum bounds. Log range bounds and sensitivity weights."*
- *"My SW-ADR sensitivity values are near zero for all parameters — the policy seems equally robust everywhere. This might be because ep_len_mean ≈ 500 (rarely failing). Should I bias sampling toward range boundaries 20% of the time to ensure sufficient boundary coverage? Implement this biased sampling in the reset() method."*
- *"Train a PPO agent on `RosPugEnvADR` with identical hyperparameters to Policy A and B. Add TensorBoard logging of: current range bounds per dimension, sensitivity weight per dimension, mean boundary return vs. interior return per dimension. Save checkpoints to `logs/ppo_rospug_c/` and final model as `policy_C_500k.zip`."*

**Testing strategy:**
- For the first 10 episodes: verify `info['dr_params']` contains per-episode values and sensitivity tracking is accumulating
- At episode 50 (first update): verify that range bounds have changed for at least one dimension and that sensitivity values differ across dimensions
- Plot TensorBoard curves `train/range_mass`, `train/range_latency`, etc. — at least one should show a steeper slope than the others, confirming differential expansion

**Exit criteria:** Policy C completes 500k steps. TensorBoard shows range evolution curves with visibly different expansion rates per parameter. Policy C walks forward in most held-out conditions at Step 7 evaluation.

**What breaks most often:** With a small episode buffer (<30 episodes), boundary variance estimates are noisy and all parameters appear equally sensitive, producing no differential expansion. Use a minimum buffer of 40–50 episodes and a dead-band (only update if `|sensitivity_d − mean_sensitivity| > threshold`) to avoid jitter.

---

##step7: Day 12 — Head-to-Head Evaluation: Policy A vs Policy B vs Policy C

**Goal:** Quantitatively demonstrate that (1) DR improves generalisation over fixed-physics training, and (2) SW-ADR improves over fixed-range DR on held-out conditions — establishing the novel contribution empirically.

**Held-out conditions (5 total — none seen exactly during B or C training):**

| # | Condition | Parameter settings | Why held-out |
|---|-----------|-------------------|-------------|
| 1 | Light body | mass_frac = 0.80 | Below B/C range [0.85, 1.15] |
| 2 | Heavy body | mass_frac = 1.20 | Above B/C range |
| 3 | High latency | latency = 15 ms | Above B/C ceiling of 10 ms |
| 4 | Slippery surface | cfm = 0.001 | Above B/C ceiling of 0.0005 |
| 5 | Worst-case combo | mass_frac = 1.20 + latency = 15 ms | Outside on two axes simultaneously |

**Protocol:** 20 deterministic episodes per policy per condition = 300 total evaluation episodes.

**Copilot prompts:**
- *"Write an evaluation script `evaluate_all.py` that accepts a list of policy checkpoints and a list of `RosPugEnvDR` configurations. For each combination, run 20 deterministic episodes and record mean displacement, fall rate, and mean episode length. Output a summary table with mean ± std and generate a grouped bar chart (3 groups per condition: A=blue, B=orange, C=green) with error bars."*
- *"Generate two subplots: (1) mean x-displacement grouped by held-out condition with ±1σ error bars, (2) fall rate grouped by held-out condition. Save as `results/comparison_chart.png`. Add a horizontal dashed line at Policy A's fixed-physics performance as a reference."*
- *"Compute statistical significance (Welch's t-test) between Policy B and Policy C mean displacement for each held-out condition. Report p-values in the summary table."*

**Testing strategy:** This step is the deliverable — the comparison table and chart. Expected pattern: Policy A degrades sharply on conditions outside its fixed training physics. Policy B holds more consistent displacement. Policy C holds as well as B or better, particularly on whichever dimensions its sensitivity curriculum expanded most.

**Exit criteria:** A summary table comparing all 3 policies × 5 conditions, and `results/comparison_chart.png`. If C does not beat B on all conditions, report honestly — a mixed result (C better on some, B better on others) is a valid finding that informs future work on sensitivity threshold tuning.

---

##step8: Day 13 — Deploy Policy B and Policy C to Real ROSPug

**Goal:** Qualitative sim-to-real validation. Run Policy B and Policy C on the physical ROSPug on at least two surfaces not used in simulation training (e.g., hard floor + carpet, or slight incline). Compare fall rate and visible gait quality against the original fixed-gait controller as a baseline.

**Why this matters for the novel claim:** Showing that a policy trained with SW-ADR transfers to real hardware with measurably different stability than fixed-range DR or no DR makes the contribution concrete, not purely simulation-based.

**Copilot prompts:**
- *"Write a ROS1 node `deploy_policy.py` that loads a Stable-Baselines3 PPO checkpoint (path from CLI argument), subscribes to `/pug/joint_states` and reads body orientation from a real IMU topic (or from a motion capture system), builds the same 26D observation vector as `RosPugEnv`, runs the policy at 50 Hz, and publishes to the 12 joint position controller command topics. Include a 3-second arming countdown and a watchdog that stops all joints if `|roll| > 0.7 rad`."*
- *"What real-hardware safety measures should I add: per-joint velocity clamp to limit max speed between commands, emergency stop on keyboard interrupt that publishes stand pose before shutting down, and a maximum runtime flag (--max-seconds 30) that gracefully stops the policy."*

**Safety protocol — follow this order:**
1. First run: support string attached overhead, 2-second burst only
2. Confirm joints respond and robot doesn't collapse immediately
3. Run 5× 2-second bursts, observe gait pattern
4. If stable: extend to 10-second runs on hard floor
5. Then test carpet and incline after hard floor passes

**Metrics to record (manual observation + video):**
- Distance walked before first fall (for each policy, 3 trials per surface)
- Whether gait is visually coordinated (legs move in diagonal pairs)
- Whether Policy C shows different behaviour from Policy B on the harder surfaces

**Exit criteria:** At least one policy (B or C) walks forward on one untrained surface without falling for a full 5-second run, where the fixed-gait controller falls or needs manual correction on the same surface.

**If you don't reach this step:** Report Policy B/C simulation results from Step 7 as the main result, and describe the real-hardware deployment architecture as "validated design, not yet physically tested" — this is a legitimate engineering contribution.

---

##step9: Day 14 — Write-Up

**Goal:** A technical report (4–6 pages) covering all three policies, the SW-ADR novel contribution, Step 7 results, and Step 8 real-hardware results (or the deployment design if Step 8 was not reached).

**Structure:**
1. **Introduction** — quadruped locomotion on budget hardware, motivation for DR and its limitations, SW-ADR as the proposed solution
2. **Related work** — OpenAI Dactyl (ADR), ANYmal sim-to-real, RMA (implicit system ID) — and how this work differs
3. **Method** — gait-residual control, `RosPugEnv`, Policy A/B/C training setup, SW-ADR mechanism with sensitivity formula
4. **Results** — Step 7 comparison table and chart; Step 8 real-hardware results or deployment design
5. **Discussion** — which DR dimension SW-ADR hardened most and why, limitations (Gazebo 9 physics fidelity, no ground friction randomisation), future work (reality-gap estimator, learned gait reference)
6. **Conclusion** — novel claim restated: SW-ADR on budget hardware achieves better held-out robustness than fixed-range DR

**Copilot prompts:**
- *"Help me structure a 4–6 page technical report on this project. The novel contribution is Sensitivity-Weighted ADR — a method where per-parameter return variance directs differential curriculum expansion in domain randomisation. Give me a section skeleton and a list of figures: training curves, range evolution curves, A/B/C comparison chart, and one real-hardware photo or video frame."*
- *"Write a related work paragraph comparing our SW-ADR to: (1) Akkaya et al. 2019 ADR (aggregate success rate, not per-parameter), (2) RMA Kumar 2021 (adaptation module, not curriculum), (3) ANYmal sim-to-real (expensive hardware). Emphasise the platform novelty (budget hardware) and the algorithmic novelty (sensitivity weighting)."*

##step10 : Day 14 — Buffer

Keep this entirely free. Something in days 5–11 will very likely take longer than planned — that's normal for a first RL project, not a sign you're behind in a way that matters.

---

## If you fall behind: what to cut, in order
1. Cut Step 8 (real-hardware deployment) first — it strengthens the paper but the Step 7 simulation results stand on their own.
2. If Policy C (SW-ADR) won't converge in time, report the sensitivity tracking mechanism and range evolution as a design contribution even without full 500k training — a documented partial result with the novel mechanism described is publishable.
3. If Step 7 evaluation can't cover all 5 held-out conditions, cut to 3 — heavy body, high latency, worst-case combo are the most discriminating.
4. Never cut the A vs B vs C comparison table — that is the core result.
