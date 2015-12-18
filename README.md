## ContractWiser Analytical Engine

### Github

git clone git@github.com:ronrlin/cw-engine.git cw-engine/

### Installing AgreementCorpus and data (still in beta)
* This routine will pull a tarball from AWS S3 and unpack it to the /data directory.

> python
> import config
> config.retrieve_and_load_data()

### Installing Flask & Libraries

sudo apt-get install python-mysqldb

pip install -r requirements.txt -t lib

### Installing the cw-engine

# clone the repository
git clone git@github.com:ronrlin/cw-engine.git cw-engine/

# rename the settings default file
cp settings-init.ini to settings.ini

# customize the settings.ini

# bootstrap data system
> python
import helper
helper.create_universe()

import provision
provision.create_system()

import statistics
statistics.compute_classified_stats()
statistics.compute_contract_group_info()
statistics.compute_provision_group_info()

### Running Tika

java -jar /home/obironkenobi/Projects/cw-engine/tika/tika-server-1.10.jar --port=8984


### Feedback
Star this repo if you found it useful. Use the github issue tracker to give
feedback on this repo.

## Contributing changes
See [CONTRIB.md](CONTRIB.md)

## Licensing
See [LICENSE](LICENSE)

## Author
Ron Lin