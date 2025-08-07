import shutil
import time
from pathlib import Path

path = Path(__file__).parent.absolute()
save_path = path.joinpath('123.zip')
print(path)

t1 = time.time()
shutil.make_archive(str(save_path.with_suffix('')), 'zip', r"C:\Code\ML\Project\StitchImageServer\temp\Input\_250801_1043_0001")
t2 = time.time()
print(t2-t1)
print('end')
