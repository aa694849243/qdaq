import subprocess
import os
with open(os.devnull, "wb") as limbo:
        for n in [30,31,109,202]:
                ip="192.168.2.{}".format(n)
                result=subprocess.Popen(["ping", "-n", "1", "-w", "2", ip],
                        stdout=limbo, stderr=limbo).wait()
                if result:
                        print (ip, "inactive")
                        print(result)
                else:
                        print (ip, "active")