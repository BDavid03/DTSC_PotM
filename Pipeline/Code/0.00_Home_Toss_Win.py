import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# 1. Set up the DataFrame with numerator and denominator data
data = {
    'League': ['BBL', 'WBBL'],
    'Win_Num': [237,170 ],
    'Win_Den': [247, 175],
    'Home_Num': [113, 100],
    'Home_Den': [247,175 ],
    'Toss_Num': [137, 91],
    'Toss_Den': [247,175 ]
}
df = pd.DataFrame(data).set_index('League')

# 2. Calculate the proportions (the bar heights)
df['Win'] = df['Win_Num'] / df['Win_Den']
df['Home'] = df['Home_Num'] / df['Home_Den']
df['Toss'] = df['Toss_Num'] / df['Toss_Den']


# 3. Set up variables for plotting
leagues = df.index
win_shares = df['Win']
home_shares = df['Home']
toss_shares = df['Toss']

x = np.arange(len(leagues))  # the label locations
width = 0.25  # the width of the bars

# 4. Create the plot
fig, ax = plt.subplots(figsize=(10, 8)) # Increased height for more label space
rects1 = ax.bar(x - width, win_shares, width, label='Win', color='#4E79A7', edgecolor='black', linewidth=0.5)
rects2 = ax.bar(x, home_shares, width, label='Home', color='#F28E2B', edgecolor='black', linewidth=0.5)
rects3 = ax.bar(x + width, toss_shares, width, label='Toss', color='#59A14F', edgecolor='black', linewidth=0.5)

# 5. Add labels, title, and formatting
ax.set_ylabel('Influence (Share of PotM Awards)', fontsize=12)
ax.set_title('PotM: Context Factors', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(leagues, fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Format the y-axis to show percentages
ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
ax.set_ylim(0, 1.2) # Adjust y-limit for annotations

# 6. Attach a text label above each bar with the fraction and percentage
def autolabel_with_fractions(rects, numerators, denominators):
    """Attach a text label above each bar, displaying its fraction and height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        num = numerators.iloc[i]
        den = denominators.iloc[i]
        
        # Create a two-line annotation
        annotation_text = f"{num}/{den}\n({height:.1%})"
        
        ax.annotate(annotation_text,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=9,
                    linespacing=1.1)

autolabel_with_fractions(rects1, df['Win_Num'], df['Win_Den'])
autolabel_with_fractions(rects2, df['Home_Num'], df['Home_Den'])
autolabel_with_fractions(rects3, df['Toss_Num'], df['Toss_Den'])

fig.tight_layout()
plt.show()