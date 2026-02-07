# StarCraft II Window Size Configuration

When running matches in realtime mode (`--realtime`), the window size is controlled by StarCraft II's client settings.

## Method 1: In-Game Settings (Recommended)

1. Open StarCraft II manually
2. Go to **Options → Graphics**
3. Configure:
   - **Display Mode**:
     - Windowed (recommended for watching bot matches)
     - Fullscreen Windowed (fills screen but can switch apps easily)
     - Fullscreen (takes over entire screen)
   - **Resolution**:
     - 1920x1080 (Full HD - good default)
     - 2560x1440 (QHD - if you have a larger monitor)
     - 1280x720 (HD - smaller but still visible)
4. Click **Apply** and close StarCraft II

These settings will persist for all future bot matches!

## Method 2: Edit Configuration File

StarCraft II stores settings in:
```
~/Library/Application Support/Blizzard/StarCraft II/Variables.txt
```

You can edit this file to set:
```
width=1920
height=1080
fullscreen=0  # 0 = windowed, 1 = fullscreen
```

After editing, save the file and restart SC2.

## Method 3: Run Multiple Windows

To watch two bots playing against each other in separate windows:

1. Set SC2 to **Windowed** mode (not fullscreen)
2. Set resolution to something smaller like **1280x720**
3. When running realtime matches, two SC2 windows will open
4. Arrange them side-by-side on your screen

Each window represents one bot's perspective!

## Tips

- **For watching single matches**: Use Fullscreen Windowed at native resolution
- **For watching tournaments**: Use smaller Windowed resolution so you can see both bots + terminal output
- **For fastest testing**: Don't use `--realtime` at all (runs headless, much faster)

## Current Settings

To see your current SC2 settings:
```bash
cat ~/Library/Application\ Support/Blizzard/StarCraft\ II/Variables.txt | grep -E "(width|height|fullscreen)"
```

## Testing

After changing settings, test with:
```bash
cd ~/programming/ai-starcraft
source .venv/bin/activate
python scripts/run_match.py --realtime
```

The game window should now use your new size settings!
