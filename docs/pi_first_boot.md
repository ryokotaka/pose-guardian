# Raspberry Pi 5 First Boot

This project uses Raspberry Pi OS Bookworm Lite (64-bit) for the first Pi
deployment. The repository currently supports Python `>=3.10,<3.12`, so
Bookworm's Python 3.11 keeps the Pi environment aligned with the development
environment.

Do not commit Wi-Fi credentials, passwords, private SSH keys, internal IP
addresses, or local benchmark clips.

## 1. Prepare the Mac-side repository

Before cloning on the Pi, confirm that the Pi dependency group is committed and
pushed:

```bash
cd ~/dev/edge-inference-guardian
git status --short
git log -1 --oneline
```

The `pi` dependency group in `pyproject.toml` should contain
`ai-edge-litert>=2.1`.

## 2. Write Raspberry Pi OS to the microSD card

In Raspberry Pi Imager:

1. Choose device: Raspberry Pi 5.
2. Choose OS: Raspberry Pi OS (other) -> Raspberry Pi OS (Legacy, 64-bit) ->
   Raspberry Pi OS (Legacy) Lite.
3. Confirm that the selected image is Bookworm, Lite, and 64-bit.
4. Choose the microSD card.
5. Configure:
   - hostname: `edge-pi`
   - admin username: your chosen username
   - password: a strong local password
   - Wi-Fi: your local SSID and password
   - localisation: your timezone and keyboard layout
   - remote access: enable SSH and use public-key authentication
6. Write and verify the image.

The default current Raspberry Pi OS image may be newer than Bookworm. Select
the Legacy Bookworm image intentionally.

## 3. Connect the hardware

1. Insert the microSD card into the powered-off Pi.
2. Connect the USB UVC webcam to a blue USB 3.0 port.
3. Connect the official 5V/5A power supply last.
4. Wait a few minutes for the first boot.

## 4. Connect over SSH

From the Mac:

```bash
ssh <pi-user>@edge-pi.local
```

If mDNS resolution does not work, find the Pi address from the router's DHCP
client list and connect using `ssh <pi-user>@<pi-ip-address>`.

## 5. Install operating-system packages

On the Pi:

```bash
sudo apt update
sudo apt full-upgrade -y
sudo apt install -y python3-venv python3-pip git curl v4l-utils ffmpeg libgl1
sudo reboot
```

Reconnect over SSH after the reboot.

## 6. Check the Pi and USB camera

On the Pi:

```bash
python3 --version
vcgencmd measure_temp
vcgencmd get_throttled
v4l2-ctl --list-devices
```

Expected baseline:

- `python3 --version` reports Python 3.11.
- `vcgencmd measure_temp` returns a temperature.
- `vcgencmd get_throttled` returns `throttled=0x0` after a clean boot.
- `v4l2-ctl --list-devices` lists the USB camera.

Record the actual `/dev/videoN` node. Do not assume `/dev/video0`.

## 7. Configure read-only GitHub access

The Pi needs its own credential to clone this private repository. The Mac-to-Pi
SSH login key is a different credential.

Generate a dedicated deploy key on the Pi:

```bash
ssh-keygen -t ed25519 -f ~/.ssh/id_github -C "edge-pi deploy key"
cat ~/.ssh/id_github.pub
```

In GitHub, open this repository's Settings -> Deploy keys -> Add deploy key.
Paste only the public key, and leave write access disabled.

Add the SSH client configuration on the Pi:

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

Test the connection:

```bash
ssh -T git@github.com
```

On the first connection, verify that GitHub's Ed25519 host fingerprint is:

```text
SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU
```

GitHub reports that it does not provide shell access. That is expected.

Clone the repository:

```bash
cd ~
git clone git@github.com:<github-user>/edge-inference-guardian.git
cd edge-inference-guardian
```

Alternative: if you do not want to store a GitHub deploy key on the Pi, copy
the working tree from the Mac with `rsync`. Do not store a personal access
token as plaintext on the Pi.

## 8. Create the Pi Python environment

On the Pi:

```bash
cd ~/edge-inference-guardian
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pi]"
./models/download_models.sh
```

## 9. Run smoke checks

On the Pi with the virtual environment active:

```bash
cd ~/edge-inference-guardian
source .venv/bin/activate

ls -la models/*.tflite
python -c "from ai_edge_litert.interpreter import Interpreter; print('litert ok')"
python -c "import cv2; print(cv2.__version__)"
python -c "from src.pose_estimator import PoseEstimator; estimator = PoseEstimator(); print(estimator.get_model_info())"
```

Importing `PoseEstimator` alone is not enough. Runtime selection is deferred, so
construct `PoseEstimator()` to confirm that LiteRT and both model files work.

## 10. Sync Mac-side changes after the first boot

Treat the Mac repository as the code source of truth. After committing and
pushing a Mac-side change, update the Pi:

```bash
cd ~/edge-inference-guardian
git pull --ff-only
```

If the Pi was provisioned with `rsync`, run the same `rsync` command again
instead. Avoid editing the same source files independently on both machines.

## Troubleshooting

### `edge-pi.local` does not resolve

Use the Pi address from the router's DHCP client list.

### The camera is missing

Run:

```bash
lsusb
v4l2-ctl --list-devices
```

Try another USB port and confirm that the official 5V/5A supply is connected.

### `import cv2` fails with a shared-library error

Install the missing graphical runtime library and retry:

```bash
sudo apt install -y libgl1
```

### `git clone` reports `Permission denied (publickey)`

Confirm that:

- `~/.ssh/id_github.pub` was added as this repository's deploy key.
- write access remains disabled.
- `~/.ssh/config` points to `~/.ssh/id_github`.
- `ssh -T git@github.com` reaches GitHub.

### `PoseEstimator()` fails

Check each layer separately:

```bash
python3 --version
python -c "from ai_edge_litert.interpreter import Interpreter; print('litert ok')"
ls -la models/*.tflite
python -c "import cv2; print(cv2.__version__)"
```
