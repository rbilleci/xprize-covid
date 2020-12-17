cd ..
rm work.tar.gz
rm *.log
tar --exclude=./venv --exclude=./.idea --exclude=work.tar.gz --exclude-vcs  -zcvf work.tar.gz .
tar -tf work.tar.gz