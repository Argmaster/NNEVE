1. Open VSCode in your repository folder

2. Use `Ctrl+Shift+P` to open command entry

3. Enter `>Python: Select Interpreter` and click `Enter`

   ![Image title](img/select_interpreter.png)

4. Select option "Enter interpreter path..."

   ![Image title](img/enter_interpreter_path.png)

5. Select option "Find..."

   ![Image title](img/find_path.png)

6. Navigate to

   - Windows: `NNEVE/.tox/devenv/Scripts/python.exe`

   - Linux: `NNEVE/.tox/devenv/bin/python`

7. Select file `python.exe`

8. Use `Ctrl+Shift+P` to open command entry

9. Enter `>Developer: Reload Window` and click `Enter`

   ![Image title](img/reload_window.png)

10. Open terminal by pulling from edge of status bar

    ![Image title](img/open_terminal_s.png)

11. Use trash can icon to kill it

    ![Image title](img/trash.png)

12. Repeat step 11.

13. Now you should have `(devenv)` prefix before your command prompt

    ![Image title](img/terminal.png)

14. If you don't have this prefix, manually activate environment:

    - Windows: `".tox/devenv/Script/activate"`

    - Linux: `source ".tox/devenv/bin/activate"`

    - for more see [this](https://docs.python.org/3/tutorial/venv.html)
