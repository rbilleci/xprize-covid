find * -not -path "predictions*" -not -path bootstrap.sh -not -path work.tar.gz -not -path sandbox_update.sh -not -path sandbox_package.sh -delete
tar xvfz work.tar.gz --exclude=sandbox_update.sh
