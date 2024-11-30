import streamlit as st
import requests

# Function to fetch Aadhar centers based on city
def get_aadhar_centers_ui(city):
    try:
        response = requests.get(f"http://aadhar-appointment-stub.onrender.com/get_aadhar_centers?city={city}")
        response_data = response.json()
        if response.status_code == 200:
            return response_data.get("aadhar_centers", [])
        else:
            st.error(response_data.get("error", "Failed to fetch Aadhar centers"))
            return []
    except Exception as e:
        st.error(f"Failed to connect to the server: {e}")
        return []

# Function to make API request for appointment booking
def book_appointment_ui(payload):
    try:
        # Ensure 'city' is valid and not "Select a city"
        if payload['city'] == "Select a city":
            return "Error: Please select a valid city."
        
        response = requests.post("http://aadhar-appointment-stub.onrender.com/book_appointment", json=payload)
        response_data = response.json()
        if response.status_code == 200:
            appointment_date = response_data.get("appointment_date", "Not specified")
            aadhar_center = payload['aadhar_center']
            city = payload['city']
            return (f"Appointment successfully booked for {payload['name']}. "
                    f"Your appointment is scheduled for {appointment_date}. "
                    f"Your appointment is at the {aadhar_center} Aadhar center in {city}.")
        return f"Error: {response_data.get('error', 'Unknown error occurred')}"
    except Exception as e:
        return f"Failed to connect to the server: {e}"

# Function to fetch appointment status
def fetch_status_ui(mobile_number):
    try:
        response = requests.get(f"http://aadhar-appointment-stub.onrender.com/appointment_status?mobile_number={mobile_number}")
        response_data = response.json()
        if response.status_code == 200:
            name = response_data.get('name', 'N/A')
            appointment_date = response_data.get('appointment_date', 'Not scheduled')
            city = response_data.get('city', 'Not specified')
            aadhar_center = response_data.get('aadhar_center', 'Not specified')
            
            return (f"Hello {name}, your appointment is scheduled for {appointment_date}. "
                    f"Your appointment is at the {aadhar_center} Aadhar center in {city}.")
        return f"Error: {response_data.get('error', 'No appointment found for this mobile number')}"
    except Exception as e:
        return f"Failed to connect to the server: {e}"

# Initialize session state variables
def initialize_session_state():
    if 'active_button' not in st.session_state:
        st.session_state.active_button = "book"
    default_values = {
        'name': "", 'mobile_number': "", 'otp': "", 'address': "", 'city': "Select a city", 
        'aadhar_center': "Select a center", 'status_number': "", 'appointment_message': "", 'appointment_confirmed': False
    }
    for key, value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Streamlit UI
def main():
    st.title("Aadhar Appointment Booking and Status Check")
    initialize_session_state()

    # Set up buttons side by side
    col1, col2 = st.columns(2)
    with col1:
        book_button = st.button("Book Appointment", key="book")
    with col2:
        status_button = st.button("Check Appointment Status", key="status")

    # Reset form if switching to book appointment or status check
    if book_button:
        st.session_state.active_button = "book"
        st.session_state.appointment_confirmed = False
        st.session_state.appointment_message = ""
    elif status_button:
        st.session_state.active_button = "status"
        st.session_state.status_number = ""

    # Book Appointment form
    if st.session_state.active_button == "book":
        if st.session_state.appointment_confirmed:
            st.success(st.session_state.appointment_message)
        else:
            st.subheader("Book your Aadhar Appointment")
            name = st.text_input("Name", value=st.session_state.name)
            mobile_number = st.text_input("Mobile Number", value=st.session_state.mobile_number)
            otp = st.text_input("OTP", value=st.session_state.otp)
            address = st.text_area("Address", value=st.session_state.address)

            # Update the city selection and store it in session state
            city = st.selectbox("City", ["Select a city", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"], 
                                index=["Select a city", "Mumbai", "Delhi", "Bangalore", "Kolkata", "Chennai"].index(st.session_state.city))
            st.session_state.city = city

            # Fetch Aadhar centers only if city is selected
            if city != "Select a city":
                aadhar_centers = get_aadhar_centers_ui(city)
                aadhar_center = st.selectbox("Aadhar Center", ["Select a center"] + aadhar_centers, 
                                            index=aadhar_centers.index(st.session_state.aadhar_center) if st.session_state.aadhar_center != "Select a center" else 0)
            else:
                aadhar_center = None

            if st.button("Book Appointment Now"):
                # Check for form validation
                if not all([name, mobile_number, otp, address, city != "Select a city", aadhar_center != "Select a center"]):
                    st.error("Please fill all the fields.")
                else:
                    payload = {"name": name, "mobile_number": mobile_number, "otp": otp, "address": address, "city": city, "aadhar_center": aadhar_center}

                    # Make the request to the backend
                    response = book_appointment_ui(payload)

                    if "successfully" in response:
                        st.session_state.appointment_confirmed = True
                        st.session_state.appointment_message = response
                        st.experimental_rerun()
                    else:
                        st.error(response)


    # Check Appointment Status form
    elif st.session_state.active_button == "status":
        st.subheader("Check your Aadhar Appointment Status")
        status_number = st.text_input("Enter your Mobile Number to check status", value=st.session_state.status_number)
        
        if st.button("Check Status"):
            if not status_number:
                st.error("Please enter your mobile number.")
            else:
                response = fetch_status_ui(status_number)
                st.write(response)

if __name__ == "__main__":
    main()
