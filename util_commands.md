### Change text in Config files
sed -i 's/old-text/new-text/g' results/_experiment_name_/*/*/cfg.yaml

### Pull from public public repo 
git remote add public https://github.com/lcswillems/torch-rl.git
git pull public master # Creates a merge commit
git push origin master
