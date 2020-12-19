cd ..
rm work.tar.gz
rm *.log
tar --exclude=./venv --exclude=./.idea --exclude=ml_train.py --exclude=work.tar.gz --exclude-vcs  -zcvf work.tar.gz .
tar -tf work.tar.gz