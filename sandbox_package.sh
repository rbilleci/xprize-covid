rm work.tar.gz
rm *.log
tar --exclude=./venv --exclude=./.git --exclude=./.idea --exclude=ml_trainer.py --exclude=work.tar.gz -zcvf work.tar.gz .
tar -tf work.tar.gz