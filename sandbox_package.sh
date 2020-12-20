rm work.tar.gz
tar --exclude __pycache__ --exclude=./venv --exclude=./.git --exclude=./.idea --exclude=ml_trainer.py --exclude=sandbox_* --exclude=work.tar.gz -zcvf work.tar.gz .
tar -tf work.tar.gz