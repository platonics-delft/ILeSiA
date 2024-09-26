import subprocess, os, threading

class Command:
    def __init__(self, cmd):
        self.cmd = cmd
        self.process = None

    def run(self, timeout):
        def target():
            self.process = subprocess.Popen(self.cmd, shell=True, stdout=open(os.devnull, 'wb'))
            self.process.communicate()

        thread = threading.Thread(target=target)
        thread.start()

        thread.join(timeout)
        if thread.is_alive():
            self.process.terminate()
            thread.join()
            raise TimeoutError("Command timed out")

        if self.process.returncode != 0:
            raise subprocess.CalledProcessError(self.process.returncode, self.cmd)


def is_roscore_running():
    command = Command("rostopic list")
    try:
        command.run(timeout=1)  # Set timeout as 1 second
    except TimeoutError as e:
        return False
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
    else:
        return True


if __name__ == '__main__':
    print(f"roscore: {'running' if is_roscore_running() else 'not running'}")