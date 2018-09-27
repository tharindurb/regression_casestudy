import pickle
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 16})

dataset='perishable_zerofilled_2016'
f = open("../data/train_{}.p".format(dataset), 'rb')
train = pickle.load(f)
f.close()

print(train.head())
print(train.shape)

# Plot unit sales of each product family
plt.figure()
ax = train.groupby(['family'])['unit_sales'].sum().plot.bar(figsize=(10,15))
ax.set_xlabel('Unit sales')
# plt.show()
plt.savefig(
    "../results/images/sum_sales_vs_family.png",
    bbox_inches='tight', dpi=500)
plt.close()

# Plot average unit sales against month for each product family
plt.figure()
ax = train.groupby(['family','month'])['unit_sales'].mean().unstack(0).plot(figsize=(20,10),linewidth=3.0)
ax.set_xticks(range(1,13))
ax.set_xticklabels(['January','February','March','April','May','June','July','August','September','October','November','December'])
ax.set_xlabel('Month')
ax.set_ylabel('Average unit sales')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
# plt.show()
plt.savefig(
    "../results/images/month_vs_family.png",
    bbox_inches='tight', dpi=500)
plt.close()

# Plot average unit sales against day of week for each product family
plt.figure()
ax = train.groupby(['family','day_of_week'])['unit_sales'].mean().unstack(0).plot(figsize=(20,10),linewidth=3.0)
ax.set_xlabel('Date of Week')
ax.set_xticks(range(0,7))
ax.set_xticklabels(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
ax.set_ylabel('Average unit sales')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
# plt.show()
plt.savefig(
    "../results/images/day_of_week_vs_family.png",
    bbox_inches='tight', dpi=500)
plt.close()
