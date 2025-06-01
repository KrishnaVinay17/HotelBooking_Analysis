import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = r"C:\Users\K KRISHNAVINAYAKA\Downloads\Hotel Bookings.csv"
df = pd.read_csv(file_path)

# Preprocessing

df['total_stay'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])
# Section 1: Booking Timing


lead_time_summary = df['lead_time'].describe()
monthly_bookings = df['arrival_date_month'].value_counts().reindex([
    'January', 'February', 'March', 'April', 'May', 'June',
    'July', 'August', 'September', 'October', 'November', 'December'
])

plt.figure(figsize=(10, 5))
sns.histplot(df['lead_time'], bins=50, kde=True, color='steelblue')
plt.title('Distribution of Lead Time (Days Before Booking)')
plt.xlabel('Lead Time (days)')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
monthly_bookings.plot(kind='bar', color='coral')
plt.title('Number of Bookings by Month')
plt.ylabel('Number of Bookings')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Section 2: Stay vs ADR

stay_vs_adr = df[df['adr'] < 500]

plt.figure(figsize=(10, 6))
sns.boxplot(x='total_stay', y='adr', data=stay_vs_adr)
plt.title('ADR vs Total Length of Stay')
plt.xlabel('Total Stay (Nights)')
plt.ylabel('Average Daily Rate (ADR)')
plt.tight_layout()
plt.show()

# # Section 3: Special Requests

plt.figure(figsize=(10, 6))
sns.countplot(x='total_of_special_requests', data=df, palette='muted')
plt.title('Total Special Requests Distribution')
plt.xlabel('Number of Special Requests')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='customer_type', y='total_of_special_requests', data=df)
plt.title('Special Requests by Customer Type')
plt.xlabel('Customer Type')
plt.ylabel('Number of Special Requests')
plt.tight_layout()
plt.show()

# # Section 4: Demographics & Patterns

top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 5))
top_countries.plot(kind='bar', color='seagreen')
plt.title('Top 10 Countries by Number of Bookings')
plt.xlabel('Country')
plt.ylabel('Number of Bookings')
plt.tight_layout()
plt.show()

cust_stay_adr = df.groupby('customer_type')[['total_stay', 'adr']].mean()

cust_stay_adr.plot(kind='bar', figsize=(10, 5))
plt.title('Average Stay Duration and ADR by Customer Type')
plt.ylabel('Average')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Section 5: Cancellation Rate by Hotel
cancel_rate_by_hotel = df.groupby('hotel')['is_canceled'].mean().sort_values()

plt.figure(figsize=(8, 5))
cancel_rate_by_hotel.plot(kind='bar', color='salmon')
plt.title('Cancellation Rate by Hotel Type')
plt.ylabel('Cancellation Rate')
plt.xlabel('Hotel')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Section 6: ADR Trends Over Time
monthly_adr = df.groupby(['arrival_date_year', 'arrival_date_month'])['adr'].mean().unstack().T
months_ordered = ['January', 'February', 'March', 'April', 'May', 'June','July', 'August', 'September', 'October', 'November', 'December']
monthly_adr = monthly_adr.loc[months_ordered]

monthly_adr.plot(figsize=(12, 6), marker='o')
plt.title('Average Daily Rate Trend Over Months by Year')
plt.ylabel('ADR')
plt.xlabel('Month')
plt.xticks(rotation=45)
plt.legend(title='Year')
plt.tight_layout()
plt.show()

# Section 7: Booking Distribution by Market Segment
plt.figure(figsize=(10, 5))
sns.countplot(x='market_segment', data=df, order=df['market_segment'].value_counts().index, palette='coolwarm')
plt.title('Bookings by Market Segment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Section 8: Repeated Guests vs New Guests
repeated_guest_stats = df['is_repeated_guest'].value_counts(normalize=True)

plt.figure(figsize=(6, 6))
plt.pie(repeated_guest_stats, labels=['New Guest', 'Repeated Guest'], autopct='%1.1f%%', startangle=140, colors=['lightblue', 'orange'])
plt.title('New vs. Repeated Guests')
plt.tight_layout()
plt.show()

# Section 9: ADR by Distribution Channel
plt.figure(figsize=(10, 6))
sns.boxplot(x='distribution_channel', y='adr', data=df[df['adr'] < 500])
plt.title('ADR by Distribution Channel')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
