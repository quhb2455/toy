
 
def pip_install(package) :
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    

def submit() :
    pip_install("tqdm")
    
    from tqdm import tqdm
    
    for i in tqdm(range(10000)) :
        continue

if __name__ == "__main__" :
    submit()
    
    