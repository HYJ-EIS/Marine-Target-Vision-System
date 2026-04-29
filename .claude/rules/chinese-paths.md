# Chinese Path Handling

Windows `cv2.imread()` / `cv2.imwrite()` uses ANSI encoding internally, which corrupts Chinese characters in file paths.

## Rule

- **NEVER** use `cv2.imread()` or `cv2.imwrite()` directly in this project
- **ALWAYS** use `cv_utils.imread_unicode()` / `cv_utils.imwrite_unicode()` for image I/O
- When adding new image loading code, verify it uses the Unicode-safe wrappers

## Verification

Search for bare `cv2.imread` calls — any match outside `cv_utils.py` is a bug:
```bash
grep -rn "cv2\.imread\b" --include="*.py" | grep -v cv_utils
```
