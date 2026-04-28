import subprocess

def get_git_log():
    try:
        output = subprocess.check_output(['git', '--no-pager', 'log', '-p', 'munl/settings.py']).decode('utf-8', errors='ignore')
        
        commits = output.split('\ncommit ')
        for commit in commits:
            if not commit.startswith('commit'):
                commit = 'commit ' + commit
            
            lines = commit.split('\n')
            commit_hash = lines[0].replace('commit ', '').strip()
            date = ""
            for line in lines:
                if line.startswith('Date:'):
                    date = line
                    break
            
            # Print if any line contains HP_POS_FLOAT
            if any('HP_POS_FLOAT' in l for l in lines):
                print(f"--- Commit {commit_hash[:8]} ---")
                print(date)
                for i, l in enumerate(lines):
                    if 'HP_POS_FLOAT' in l:
                        # Print context
                        start = max(0, i-2)
                        end = min(len(lines), i+4)
                        for j in range(start, end):
                            print(lines[j])
                        print('---')
                    
    except Exception as e:
        print(e)

if __name__ == "__main__":
    get_git_log()
