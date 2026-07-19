Output demos/final.gif
Set FontSize 18
Set Width 900
Set Height 500
Set Padding 24
Set Margin 20
Set MarginFill "#1e1e2e"
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set TypingSpeed 60ms
Set Framerate 30

Env FORCE_COLOR "1"

Hide
Type "cd /Users/akshath/dev/weightlens && clear" Enter
Sleep 1s
Show

Sleep 500ms
Type "ls -lh demo/checkpoints/corrupted_spike.pth" Enter
Sleep 3s

Ctrl+L
Sleep 200ms
Type "uv run lens analyze demo/checkpoints/corrupted_spike.pth" Enter
Sleep 5s
Sleep 3s
Ctrl+L
Sleep 200ms
Type "pip install weightlens"
Sleep 2s
