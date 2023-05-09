from pathlib import Path

class rst_file:

    def __init__(self, path):
        self.path = path

    def read(self):
        with open(self.path, 'r') as f:
            self.lines = f.readlines()

    def write(self):
        with open(self.path, 'w') as f:
            f.writelines(self.lines)

class Process:

    def __init__(self):
        pass

    def edit_title(self, rst):
        title_line = rst.lines[0]
    
        if 'module' in title_line:
            title_line = title_line.replace('module', '')

        elif 'package' in title_line:
            title_line = title_line.replace('package', '')

        if 'agox.module.rst' in rst.path:
            title_line = 'module \n'

        # Want the title to just be the last part of the name of the file
        title_line = title_line.split('.')[-1]

        rst.lines[0] = title_line


if __name__ == '__main__':
    from argparse import ArgumentParser
    import glob

    parser = ArgumentParser()
    parser.add_argument('-p', '--path', type=str)
    args = parser.parse_args()

    path = Path(args.path)

    files = glob.glob(args.path + '*.rst')

    print(f'Files to process: {len(files)}')

    p = Process()

    for file in files:
        rst = rst_file(file)
        rst.read()

        # Process: 
        p.edit_title(rst)

        # Write:
        rst.write()

    # Want to change such that there is an indent so the contents of agox.rst 
    # should go in module.rst 
    rst_modules = rst_file(path / 'modules.rst')
    rst_agox = rst_file(path / 'agox.rst')

    rst_modules.read()
    rst_agox.read()

    rst_modules.lines = rst_agox.lines
    rst_modules.lines[0] = 'API Reference \n'
    rst_modules.lines[1] = len(rst_modules.lines[0]) * '=' + '\n'
    rst_modules.write()


