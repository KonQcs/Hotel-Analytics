Synthetic Hotel Analytics Datasets
===================================
Files:
- hotel_bookings_synth.csv: 15,000 bookings across 15 synthetic hotels (2024-01-01 to 2025-12-31).
- hotel_occupancy_daily.csv: Derived daily occupancy per hotel.

Key columns (bookings):
- booking_id: unique id
- hotel_id, room_id
- created_at, checkin_date, checkout_date
- lead_time_days, length_of_stay, num_guests
- channel: Direct, OTA-Booking, OTA-Expedia, Phone, Corporate
- cancellation_policy: NRF, Flexible, Semi-Flex
- price_per_night, total_price
- refundable: 1/0
- is_cancelled: 1/0
- cancel_date: null if not cancelled
- weather_score: 0 (bad) to 1 (great)
- events_nearby_level: 0 none, 1 some, 2 major
- competitor_price_index: around 1.0
- holiday_flag: 1 on select Greek holidays
- created_dow, checkin_dow: day of week names

Key columns (occupancy):
- date
- occupied_rooms, capacity_rooms, occupancy_rate
- dow, holiday_flag

Notes:
- Seasonality approximated; cancellations depend on lead time, policy, and channel.
- Occupancy excludes stays cancelled before check-in.
- For coursework use only.
