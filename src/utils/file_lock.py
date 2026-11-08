"""File Lock Utility - 文件锁管理工具

提供跨平台的文件锁机制，支持：
- 文件锁获取与释放
- 超时自动清理僵尸锁
- 带重试的文件读写操作
- 启动时清理残留锁文件

参考经验：
- 病历质控系统文件锁冲突问题处理
- 僵尸锁超时自动覆盖机制
"""
import json
import os
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional


class FileLockError(Exception):
    """文件锁操作异常"""
    pass


class FileLock:
    """基于文件的锁机制"""
    
    def __init__(self, lock_file: Path, timeout: float = 30.0, check_interval: float = 0.5, stale_timeout: float = 300.0):
        """
        Args:
            lock_file: 锁文件路径
            timeout: 获取锁的最大等待时间（秒）
            check_interval: 检查锁状态的间隔（秒）
            stale_timeout: 僵尸锁超时时间（秒），超过此时间自动清理
        """
        self._lock_file = lock_file
        self._timeout = timeout
        self._check_interval = check_interval
        self._stale_timeout = stale_timeout
        self._pid = os.getpid()
    
    def _is_stale(self) -> bool:
        """检查锁文件是否为僵尸锁"""
        if not self._lock_file.exists():
            return False
        
        try:
            stat = self._lock_file.stat()
            age = time.time() - stat.st_mtime
            return age > self._stale_timeout
        except Exception:
            return False
    
    def _acquire(self) -> bool:
        """尝试获取锁（非阻塞）"""
        try:
            fd = os.open(str(self._lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{self._pid}\n".encode('utf-8'))
            os.close(fd)
            return True
        except OSError:
            return False
    
    def acquire(self) -> bool:
        """获取锁（带超时）"""
        start_time = time.time()
        
        while time.time() - start_time < self._timeout:
            if self._is_stale():
                self._force_release()
            
            if self._acquire():
                return True
            
            time.sleep(self._check_interval)
        
        return False
    
    def release(self) -> None:
        """释放锁"""
        try:
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception:
            pass
    
    def _force_release(self) -> None:
        """强制释放锁（用于清理僵尸锁）"""
        try:
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception:
            pass
    
    def __enter__(self) -> 'FileLock':
        if not self.acquire():
            raise FileLockError(f"Failed to acquire lock for {self._lock_file}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()


def cleanup_stale_locks(base_dir: Path, stale_timeout: float = 300.0) -> int:
    """清理目录中过期的锁文件
    
    Args:
        base_dir: 基础目录
        stale_timeout: 过期时间（秒）
    
    Returns:
        清理的锁文件数量
    """
    count = 0
    for lock_file in base_dir.rglob("*.lock"):
        try:
            age = time.time() - lock_file.stat().st_mtime
            if age > stale_timeout:
                lock_file.unlink()
                count += 1
        except Exception:
            pass
    return count


def write_with_lock(file_path: Path, content: Any, 
                    timeout: float = 30.0, 
                    check_interval: float = 0.5,
                    max_retries: int = 3,
                    retry_delay: float = 1.0) -> bool:
    """带锁写入文件
    
    Args:
        file_path: 文件路径
        content: 内容（dict 或 list 将被 JSON 序列化）
        timeout: 锁超时时间
        check_interval: 锁检查间隔
        max_retries: 最大重试次数
        retry_delay: 重试延迟
    
    Returns:
        是否写入成功
    """
    lock_file = file_path.parent / f"{file_path.name}.lock"
    
    for attempt in range(max_retries):
        try:
            with FileLock(lock_file, timeout, check_interval):
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if isinstance(content, (dict, list)):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(content, f, ensure_ascii=False, indent=2)
                else:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(str(content))
                
                return True
        except FileLockError:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise
        except PermissionError as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise FileLockError(f"Permission denied when writing {file_path}: {e}")
        except Exception as e:
            raise FileLockError(f"Error writing {file_path}: {e}")
    
    return False


def read_with_lock(file_path: Path, 
                   timeout: float = 10.0,
                   check_interval: float = 0.5) -> Any:
    """带锁读取文件
    
    Args:
        file_path: 文件路径
        timeout: 锁超时时间
        check_interval: 锁检查间隔
    
    Returns:
        文件内容
    """
    lock_file = file_path.parent / f"{file_path.name}.lock"
    
    try:
        with FileLock(lock_file, timeout, check_interval):
            if not file_path.exists():
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return content
    except FileLockError:
        raise
    except Exception as e:
        raise FileLockError(f"Error reading {file_path}: {e}")


def atomic_write(file_path: Path, content: Any, 
                 timeout: float = 30.0,
                 check_interval: float = 0.5,
                 max_retries: int = 10,
                 retry_delay: float = 3.0) -> bool:
    """原子写入文件（先写临时文件，再替换）
    
    - Windows 上 `shutil.move` 对已存在目标会回退 `copy2+unlink`；若目标文件正被其他
      进程占用（如 OneDrive 同步、杀软扫描）会抛出瞬时 `PermissionError`。此处带退避
      重试，避免 merge 写 master 环节因瞬时锁定而中断。
    
    Args:
        file_path: 文件路径
        content: 内容
        timeout: 锁超时时间
        check_interval: 锁检查间隔
        max_retries: 替换失败的最大重试次数
        retry_delay: 每次重试前的等待秒数
    
    Returns:
        是否写入成功
    """
    lock_file = file_path.parent / f"{file_path.name}.lock"
    temp_file = None
    
    try:
        with FileLock(lock_file, timeout, check_interval):
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            for attempt in range(max_retries):
                temp_file = file_path.with_suffix(f".tmp_{int(time.time())}")
                try:
                    if isinstance(content, (dict, list)):
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            json.dump(content, f, ensure_ascii=False, indent=2)
                    else:
                        with open(temp_file, 'w', encoding='utf-8') as f:
                            f.write(str(content))
                    
                    shutil.move(str(temp_file), str(file_path))
                    return True
                except PermissionError as e:
                    if temp_file and temp_file.exists():
                        try:
                            temp_file.unlink()
                        except OSError:
                            pass
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        continue
                    raise FileLockError(
                        f"Atomic write failed for {file_path} after {max_retries} attempts: {e}"
                    )
    except FileLockError:
        raise
    except Exception as e:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise FileLockError(f"Atomic write failed for {file_path}: {e}")