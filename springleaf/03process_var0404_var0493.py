import pandas as pd

# var0404
df = pd.read_table('var0404.dat', header=None, names=["occupation"])
df[df.occupation=="conta"] = "contact"
ocs = ["president", "manager", "director", 
       "supervisor", "owner", "member", 
       "professor", "teacher", "nurse", 
       "sales", "coordinator", "surgeon", 
       "dentis", "secretary", "contractor", 
       "medic", "agent", "treasurer",
       "pharma", "office", "partner",
       "assistant"]

for oc in ocs:
    df["occupation"] = df["occupation"].apply(lambda x: oc if oc in x else x)

df["occupation"] = df["occupation"].apply(lambda x: "ceo" if (("ceo" in x) or ("cfo" in x) or ("chief executive officer" in x) or ("chief financial officer" in x)) else x)
df["occupation"] = df["occupation"].apply(lambda x: "attorney" if (("attorney" in x) or ("lawyer" in x)) else x)
df["occupation"] = df["occupation"].apply(lambda x: "manager" if x=="manag" else x)

tab = df["occupation"].value_counts()
others = tab.index[tab<=10]
df["occupation"] = df["occupation"].apply(lambda x: "others" if x in others.values else x)
df.to_csv("var0404_proc.dat", index=False, header=False, quoting=2)
tab = df["occupation"].value_counts()
tab.to_csv("var0404_proc_table.csv", header=False)

# var0493
df = pd.read_table('var0493.dat', header=None, names=["occupation"])
ocs = ["nurs", "sales", "cosmetologist", 
       "pharma", "real estate", "accountant", 
       "physician", "dentis", "attorney", 
       "therapist", "electrician", "engineer", 
       "technician", "technologist", "contractor", 
       "guard", "doctor", "medic", 
       "broker", "denta", "manager",
       "agent"]
for oc in ocs:
    df["occupation"] = df["occupation"].apply(lambda x: oc if oc in x else x)

df["occupation"] = df["occupation"].apply(lambda x: "cosmetologist" if "barber" in x else x)
df["occupation"] = df["occupation"].apply(lambda x: "cosmetologist" if "beautician" in x else x)

tab = df["occupation"].value_counts()
others = tab.index[tab<=10]
df["occupation"] = df["occupation"].apply(lambda x: "others" if x in others.values else x)
df.to_csv("var0493_proc.dat", index=False, header=False, quoting=2)
tab = df["occupation"].value_counts()
tab.to_csv("var0493_proc_table.csv", header=False)
