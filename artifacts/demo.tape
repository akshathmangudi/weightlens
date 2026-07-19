Output artifacts/final.gif
Set FontSize 18
Set Width 900
Set Height 500
Set Padding 24
Set Margin 20
Set MarginFill "#1e1e2e"
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 50ms
Set Framerate 30

Env FORCE_COLOR "1"

Hide
Type "cd /Users/akshath/dev/weightlens && clear" Enter
Sleep 500ms
Show

Sleep 500ms
Type "ls -lh artifacts/checkpoints/corrupted_spike.pth" Enter
Sleep 2.5s

Ctrl+L
Sleep 200ms
Type "uv run lens analyze artifacts/checkpoints/corrupted_spike.pth" Enter
Sleep 4s
Sleep 2s
Ctrl+L
Sleep 200ms
Type "pip install weightlens"
Sleep 1.5s
