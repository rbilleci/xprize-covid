cd ..
find * -not -path "predictions*" -not -path bootstrap.sh -not -path work.tar.gz -not -path "bin*" -delete
tar xvfz work.tar.gz
