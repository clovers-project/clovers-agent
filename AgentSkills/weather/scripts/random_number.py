if __name__ == "__main__":
    import sys
    import time
    import random

    def progress_bar(total_seconds, fall_steps=None):
        total_steps = 100
        total_steps_d10 = total_steps // 10
        interval = total_seconds / 100
        steps = min(fall_steps, total_steps) if fall_steps else total_steps
        for i in range(1, steps + 1):
            n = i // 10
            filled = "#" * n
            empty = "." * (total_steps_d10 - n)
            bar = f"[{filled}{empty}]"
            sys.stdout.write(f"\r{bar} {i}%")
            sys.stdout.flush()
            time.sleep(interval)
        print()

    def spinner(title, end="", duration=1.5):
        """在当前位置显示转圈动画，结束后光标回到行首"""
        # chars = ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"]
        chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        start_time = time.time()
        i = 0
        while time.time() - start_time < duration:
            sys.stdout.write(f"\r{chars[i % len(chars)]} {title}")
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
        sys.stdout.write(f"\r" + " " * (len(title) + 1))
        if end:
            sys.stdout.write(f"\r{end}\n")
        sys.stdout.flush()

    print("[SYSTEM] 正在初始化参数解析模块...")
    progress_bar(3)
    print("[CORE] 正在解析空间量子参数...")
    try:
        min_val = int(sys.argv[1])
        max_val = int(sys.argv[2])
        progress_bar(5)
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        print(f"[OK] 目标区间已锁定: [{min_val}, {max_val}]")
    except Exception as e:
        min_val, max_val = 1, 10
        progress_bar(5, 28)
        print(f"[ERROR] 失败，回退到默认区间：[{min_val}, {max_val}]")
    spinner("[CALIB] 正在进行实验校准...", "[CALIB] 正在进行实验校准...", 4)
    print("[WARN] 检测到非稳态，正在尝试修复...")
    progress_bar(5)
    print("[OK] 基准值稳定。")
    time.sleep(0.5)
    for i in range(1, 3):
        spinner(f"[NO.{i}] 正在观测", f"[RETRY] 第 {i} 次观测重试...", 4)
    result = random.randint(min_val, max_val)
    spinner("[NO.3] 正在观测", f"[SUCCESS] 参数观测完成，结果为: {result}", 2)
