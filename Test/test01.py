import shutil
from pathlib import Path

path = Path(__file__).parent.absolute()
save_path = path.joinpath('123.zip')
print(path)

shutil.make_archive(str(save_path.with_suffix('')), 'zip', r"C:\Code\ML\Project\StitchImageServer\temp\output")
print('end')
