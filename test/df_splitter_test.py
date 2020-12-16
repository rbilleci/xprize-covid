import oxford_loader
import oxford_processor
import df_00_splitter

df = oxford_processor.process(oxford_loader.load('OxCGRT_latest.csv'))
sx, sy, sz = df_00_splitter.split(df, 21, 21)
