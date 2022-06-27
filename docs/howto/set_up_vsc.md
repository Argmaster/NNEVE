# HowTo set up Visual Studio Code for development

1. Make sure you have following extensions installed:

   - Python
   - Pylance
   - YAML
   - GitLens — Git supercharged
   - indent-rainbow
   - Test Explorer UI
   - Test Explorer UI

2. Use `Ctrl+Shift+P` to open VSC command prompt

3. Enter `>Preferences: Open Settings (UI)` and click `Enter` to open Settings
   tab

4. For all following use setting search field:

   1. Type `Files: Auto Save` and select option `afterDelay`

   2. Type `Python: Language Server` and select `Pylance`

   3. Type `Files: Auto Save Delay` and set value delay value to `1000`

   4. Type `Python > Analysis: Diagnostic Mode` and set value to `workspace`

   5. Type `Python > Analysis: Extra Paths` and use `Add item` to add
      `./source/`

   6. Type `Python > Formatting: Provider` and set it to `black`

   7. Type `Python > Linting: Flake8 Enabled` and check it

   8. Type `Python > Linting: Pylint Enabled` and uncheck it

   9. Type `Docstring Format` and select `numpy`

   10. Type `Python > Analysis: Type Checking Mode` and set to `basic`

   11. Type `Indent Rainbow: Colors` and check option
       `Color On Whitespace Only`:

   12. Type `Indent Rainbow: Colors` and in section `Colors` use
       `Edit in settings.json` to enter settings in json form, then use
       following to update `"indentRainbow.colors"` section:

   ```
   "indentRainbow.colors": [
       "rgba(255,255,255,0.02)",
       "rgba(255,255,255,0.04)",
       "rgba(255,255,255,0.055)",
       "rgba(255,255,255,0.07)",
       "rgba(255,255,255,0.085)",
       "rgba(255,255,255,0.10)",
       "rgba(255,255,255,0.115)"
   ]
   ```
