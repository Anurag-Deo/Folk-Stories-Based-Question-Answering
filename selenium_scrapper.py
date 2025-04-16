"""
Fairy Tale Web Scraper

This script automates the process of scraping folk stories from fairytalez.com.
It handles user authentication, navigates to story pages, and extracts story content
including title and text. The extracted stories are saved to a CSV file.

Requirements:
- Selenium WebDriver
- BeautifulSoup4
- pandas
- dotenv
- A valid fairytalez.com account
- Chrome/Brave browser and matching chromedriver
"""

import os
import time

import pandas as pd
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# Load environment variables from .env file
load_dotenv()
# Get the username and password from environment variables
USERNAME = os.getenv("USERNAME")
PASSWORD = os.getenv("PASSWORD")
# Check if the environment variables are set
if USERNAME is None or PASSWORD is None:
    raise ValueError("Please set the USERNAME and PASSWORD environment variables.")

# Constants for site URLs
LOGIN_URL = "https://fairytalez.com/login/"
STORY_URL = "https://fairytalez.com/momotaro/"


def login(driver, username, password):
    """Logs into fairytalez.com using Selenium.

    Navigates to the login page, finds the form elements, and submits
    login credentials. Includes error handling for page load issues.

    Args:
        driver (WebDriver): The Selenium WebDriver instance.
        username (str): Your fairytalez.com username/email.
        password (str): Your fairytalez.com password.

    Returns:
        bool: True if login was successful, False otherwise.
    """
    try:
        # Navigate to the login page
        driver.get(LOGIN_URL)

        # Sleep for 5 seconds to allow the page to load.
        # You should use WebDriverWait instead.
        time.sleep(5)

        # Find the username/email and password fields and the submit button.
        username_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.NAME, "email")
            )  # Form field identified by name
        )
        print(username_field)
        password_field = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.ID, "firebase-login__password")
            )  # Form field identified by ID
        )
        print(password_field)
        submit_button = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "firebase-login__submit"))
        )
        print(submit_button)
        # Clear the fields in case they have default values
        username_field.clear()
        password_field.clear()
        # Enter the credentials and submit the form
        username_field.send_keys(username)
        password_field.send_keys(password)
        submit_button.click()

    except TimeoutException:
        print("Error: Timed out waiting for login page elements.")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during login: {e}")
        return False

    # Allow time for login to complete
    time.sleep(5)
    # Refresh the page to ensure we are logged in
    driver.refresh()
    time.sleep(3)
    return True


def get_story_details(driver, url):
    """
    Fetches and extracts the story title and text from a given URL
    using an authenticated Selenium session.

    Args:
        driver (WebDriver): The Selenium WebDriver instance with active session.
        url (str): The URL of the story page to scrape.

    Returns:
        dict: A dictionary containing 'title' and 'story' keys with corresponding content.
              Returns None if extraction fails.
    """
    try:
        driver.get(url)  # Navigate to the story URL
        # Wait for the main story content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located(
                (By.CSS_SELECTOR, "section.entry.classic-tale")
            )
        )

        # Now that the page is loaded, use BeautifulSoup for HTML parsing
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Extract the story title
        title_element = soup.find("h1", class_="title entry-title")
        title = title_element.text.strip() if title_element else None

        # Extract the story content
        story_section = soup.find("section", class_="entry classic-tale")
        # Extract all paragraphs from the story section
        story_paragraphs = story_section.find_all("p") if story_section else []
        # Combine paragraphs into a single text string
        story_text = " ".join(p.text.strip() for p in story_paragraphs)

        return {"title": title, "story": story_text}

    except TimeoutException:
        print(f"Error: Timed out waiting for story page to load: {url}")
        return None
    except Exception as e:
        print(f"An error occurred while fetching the story: {e}")
        return None


def main():
    """
    Main function that orchestrates the scraping process.

    - Sets up the WebDriver
    - Authenticates with the website
    - Iterates through story URLs to extract content
    - Saves the extracted stories to a CSV file
    """
    # Login credentials for fairytalez.com
    username = USERNAME
    password = PASSWORD

    # Configure Chrome WebDriver
    service = Service(
        executable_path="/mnt/01D8D2939399DF30/Python/Capcha Automation/GoogleRecaptchaBypass/chromedriver-linux64/chromedriver"
    )
    # Set up browser options (using Brave browser)
    options = Options()
    options.binary_location = "/usr/bin/brave-browser"
    # options.add_argument("--headless")  # Uncomment to run in headless mode

    # Load the list of story URLs from CSV file
    df = pd.read_csv("Datasets/folk_tales_links.csv")
    urls = df["source"].tolist()
    urls = urls[:150]  # Limit to first 150 stories

    # Initialize the WebDriver
    driver = webdriver.Chrome(options=options, service=service)
    print("Loaded Chrome WebDriver")

    try:
        # Attempt to log in to the website
        if login(driver, username, password):
            # Open output file to save stories
            with open("Final_stories.csv", "a") as file:
                # Process each story URL
                for url in urls:
                    try:
                        STORY_URL = url
                        # Get the story details
                        story_details = get_story_details(driver, STORY_URL)
                        if story_details:
                            print(f"Title: {story_details['title']}")

                            # Save the URL, title, and story to CSV file
                            # using '###' as a delimiter between fields
                            file.write(
                                f"{STORY_URL}###{story_details['title']}###{story_details['story']}\n"
                            )
                            file.flush()  # Ensure data is written immediately
                        else:
                            print("Failed to retrieve story details.")
                    except Exception as e:
                        print(f"An error occurred while processing the story: {e}")
        else:
            print("Login failed. Could not retrieve story.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # finally:
    #     driver.quit()  # Always close the browser, even if there are errors


if __name__ == "__main__":
    main()
