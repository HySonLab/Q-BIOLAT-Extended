import pandas as pd
df = pd.read_csv("data/proteingym/gfp.csv")
df.sample(n=2000, random_state=42).to_csv("data/proteingym/gfp_2k.csv", index=False)
print("saved data/proteingym/gfp_2k.csv")
