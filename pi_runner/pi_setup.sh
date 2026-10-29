#!/usr/bin/env bash
# One-time system setup of the Trex Raspberry Pi (Pi 4, Raspberry Pi OS Bookworm 64-bit).
# Idempotent: safe to re-run. Run on the Pi as the robot user, then reboot if it says so:
#   bash pi_setup.sh            (deploy.sh copies it to ~/trex/pi_runner/)
# What it does, and why:
#   - packages: python venv/dev, i2c-tools, build tools, tmux/htop
#   - locale en_US.UTF-8 (ssh sessions from the PC ask for it)
#   - BNO055 on bit-banged I2C bus 3 (GPIO2/3), hardware I2C off: the BNO055 stretches the
#     clock and the hardware block corrupts bit 7 of bytes read
#   - serial console off, UART on (GPIO14/15 free for a future direct servo bus)
#   - CPU governor "performance" at boot (steady 25 Hz loop timing; ~+0.5 W)
# The static IP is set separately with nmcli (it depends on the network).
set -euo pipefail
F=/boot/firmware/config.txt
reboot_needed=0

sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
  python3-venv python3-dev python3-pip i2c-tools git build-essential rsync htop tmux locales >/dev/null

if ! locale -a 2>/dev/null | grep -qi '^en_US.utf8$'; then
  sudo sed -i 's/^# *en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen
  sudo locale-gen >/dev/null && sudo update-locale LANG=en_US.UTF-8
fi

if ! grep -q '^dtoverlay=i2c-gpio,bus=3' "$F"; then
  sudo cp "$F" "$F.bak-$(date +%Y%m%d-%H%M%S)"
  sudo sed -i 's/^dtparam=i2c_arm=on/dtparam=i2c_arm=off/; /^dtparam=i2c_arm_baudrate/d' "$F"
  printf '\n# BNO055 clock-stretches; the hardware I2C block corrupts bit 7. Bit-banged I2C on the same pins (GPIO2 SDA, GPIO3 SCL).\ndtoverlay=i2c-gpio,bus=3,i2c_gpio_sda=2,i2c_gpio_scl=3,i2c_gpio_delay_us=2\n' | sudo tee -a "$F" >/dev/null
  reboot_needed=1
fi
grep -qx 'i2c-dev' /etc/modules || { echo i2c-dev | sudo tee -a /etc/modules >/dev/null; reboot_needed=1; }

if grep -q 'console=serial0' /boot/firmware/cmdline.txt || ! grep -q '^enable_uart=1' "$F"; then
  sudo raspi-config nonint do_serial_cons 1
  sudo raspi-config nonint do_serial_hw 0
  reboot_needed=1
fi

if [ ! -f /etc/systemd/system/cpu-performance.service ]; then
  sudo tee /etc/systemd/system/cpu-performance.service >/dev/null <<'EOF'
[Unit]
Description=CPU governor performance (steady Trex control-loop timing)

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for g in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do echo performance > "$g"; done'

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload
  sudo systemctl enable --now cpu-performance.service
fi

groups | grep -qw dialout || { sudo usermod -aG dialout,i2c "$USER"; echo "added $USER to dialout,i2c (log in again)"; }

echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
[ "$reboot_needed" = 1 ] && echo "REBOOT NEEDED (sudo reboot)" || echo "setup complete, no reboot needed"
