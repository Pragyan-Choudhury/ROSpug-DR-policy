# ROSPug Domain Randomization Project — 2-Week Roadmap (with GitHub Copilot)

## Honest scope note (read this first)

The originally discussed project (real-hardware sim-to-real transfer + a "reality gap estimator" tuned from real trials) is a multi-month undertaking for someone starting from zero. To actually finish something real in 2 weeks, this roadmap targets a scoped, defensible version:

**Core deliverable (must finish):** Train an RL walking policy for ROSPug in Gazebo simulation two ways — (A) normal training on fixed physics parameters, (B) training with domain randomization (friction, mass, servo latency varied every episode) — then prove B is more robust than A when both are tested on *held-out* randomized conditions neither saw during training. This is a complete, honest research result on its own — it's exactly what the OpenAI Dactyl / ANYmal papers demonstrate, just on your platform.

**Stretch goal (only if days 1–10 go well):** Deploy policy B to the real ROSPug and qualitatively compare its walking stability against the fixed-gait baseline on a surface you didn't train on (a rug, a slope, etc.).

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

##step6 : Day 10 — Head-to-head evaluation (this is your core result)

**Goal:** Quantitatively show Policy B (randomized) generalizes better than Policy A (baseline) on physics settings *neither* policy trained on exactly — e.g., a friction value in between your sampled range, or slightly outside it.

**Copilot prompts:**
- *"Write an evaluation script that runs both saved policies (Policy A and Policy B) for 20 episodes each, across 5 held-out physics conditions I didn't use during training [list your held-out friction/mass values]. For each run, log: episode length, distance traveled, and whether the robot fell. Output a summary table and a bar chart comparing A vs B."*
- *"Help me write a short statistical summary (mean ± std) comparing fall rate and distance traveled between Policy A and Policy B across the held-out conditions, and generate a matplotlib bar chart for my report."*

**Testing strategy:** This step *is* the test — the deliverable is the comparison table/chart itself. Your hypothesis (and likely result): Policy A does great on the exact conditions it trained on but degrades sharply on held-out conditions; Policy B is more consistent across all of them, possibly at a small cost to peak performance on the "easy" condition.

**Exit criteria:** You have a table and chart clearly showing A vs. B fall rate / distance across held-out conditions, whichever direction the result actually goes. (If B doesn't beat A, that's still a valid, reportable finding — don't massage the numbers; report it honestly and discuss why, e.g. randomization ranges may need tuning.)

---

##step7 : Day 11–12 — Stretch goal: real ROSPug test (only if on schedule)

**Goal:** Qualitative comparison of Policy B vs. the robot's original fixed-gait controller on the real ROSPug, on a surface not used in training (rug, slight incline, etc.).

**Copilot prompts:**
- *"Help me export my trained Stable-Baselines3 policy and write a ROS1 node that loads it and publishes the same command topics the real ROSPug expects, replacing the simulated environment's action publisher with the real robot's."*
- *"What's the safest way to test a possibly-unstable RL policy on real quadruped hardware for the first time — should I use a support harness, limit max servo torque, or test in short 2–3 second bursts first?"*

**Testing strategy:** Short bursts first (as Copilot will likely also suggest) — 2–3 seconds of walking at a time before letting it run freely, so a bad policy doesn't damage the servos.

**If you don't reach this step:** that's fine — say so plainly in your write-up as "future work," rather than skipping it silently. The Day 10 result stands on its own.

---

##step8 : Day 13 — Write-up

**Goal:** A short report: problem statement, method (both policies, randomization ranges used), Day 10 results (table/chart), stretch-goal result if you got there, and an honest "limitations & future work" section (this is exactly where the reality-gap-estimator idea from your original proposal belongs — as future work, not something you're claiming to have built).

**Copilot prompts:**
- *"Help me structure a short technical report (roughly 3–5 pages) on this project: intro, related work [OpenAI Dactyl, ANYmal sim-to-real], method, results, limitations, future work. I'll fill in specifics — give me the section skeleton and a suggested figure list."*

##step9 : Day 14 — Buffer

Keep this entirely free. Something in days 5–9 will very likely take longer than planned — that's normal for a first RL project, not a sign you're behind in a way that matters.

---

## If you fall behind: what to cut, in order
1. Cut the real-hardware stretch goal first (days 11–12) — it was never required.
2. If Day 10's evaluation can't run on 5 held-out conditions, cut to 2–3 — a smaller, honest comparison beats a rushed broken one.
3. If Policy B (randomized) genuinely won't train in time, report Policy A alone plus a clear explanation of what randomization you attempted and why it didn't converge — a documented failed experiment is a legitimate research outcome, especially for a 2-week timeline.
