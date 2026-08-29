# Re-record with: vhs docs/assets/demo.tape  (needs `brew install vhs`, an installed
# lepika on PATH, and the picked model already pulled so the demo stays short).
Output docs/assets/demo.gif

Set Shell zsh
Set FontSize 14
Set Width 1080
Set Height 560
Set Padding 16
Set Framerate 15
Set TypingSpeed 50ms
Set Theme "Catppuccin Mocha"

Hide
Type "export LEPIKA_HOME=$(mktemp -d) BROWSER=true; clear"
Enter
Show

Type "lepika"
Sleep 500ms
Enter
Sleep 2s
Type "1"
Sleep 500ms
Enter
Sleep 2s
Type "3"
Sleep 500ms
Enter
Wait+Screen@90s /Ready:/
Sleep 3s
Type "lepika status"
Enter
Wait+Screen@20s /Model/
Sleep 4s
