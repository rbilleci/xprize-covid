import oxford_loader
import oxford_processor

df = oxford_processor.process(oxford_loader.load('OxCGRT_latest.csv'))
df.info()
df = oxford_processor.process(oxford_loader.load('2020-09-30_historical_ip.csv'))
df.info()
df = oxford_processor.process(oxford_loader.load('future_ip.csv'))
df.info()
