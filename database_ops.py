"""
@file database_ops.py
@brief Database operations module for the Face Recognition Access System.
@details This module handles all interactions with the MariaDB database, including
initial setup, connection management, and CRUD (Create, Read, Update, Delete)
operations for People, Doors, Permissions, and Logs.
"""

import mariadb #pip install mariadb
import sys
import os
import shutil

## Configuration dictionary for MariaDB connection.
DB_CONFIG = {
    'user': 'root',
    'password': 'face_rec',
    'host': 'localhost',
    'port': 3306
}

## Base directory for storing registered people's images and encodings.
PEOPLE_BASE_DIR = "registered_people"

def setup_database():
    """
    @brief Creates the initial database schema and tables if they do not exist.
    @details Connects to the MariaDB server and creates the 'face_rec_db' database
    and the four main tables: 'people', 'doors', 'permissions', and 'logs'.
    Sets up primary keys, foreign keys, and necessary constraints.
    """
    conn = get_db_connection(database_name=None)
    if conn is None:
        print("Can't connect to the database server. Exiting setup.", file=sys.stderr)
        return

    try:
        cursor = conn.cursor()
        # Create the database if it doesn't exist
        cursor.execute("CREATE DATABASE IF NOT EXISTS face_rec_db")
        cursor.execute("USE face_rec_db")

        # Create 'people' table
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS people (
                person_id INTEGER PRIMARY KEY NOT NULL AUTO_INCREMENT,
                name VARCHAR(255) NOT NULL,
                age INTEGER,
                cc_num VARCHAR(50) NOT NULL,
                contact_num VARCHAR(50) NOT NULL,
                email VARCHAR(255) NOT NULL,
                company_name VARCHAR(255),
                images_path VARCHAR(255) NOT NULL,
                encodings_path VARCHAR(255) NOT NULL
            );
        """)

        # Create 'doors' table
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS doors (
                door_id INTEGER PRIMARY KEY NOT NULL AUTO_INCREMENT,
                location VARCHAR(255) NOT NULL,
                description VARCHAR(255)
            );
        """)

        # Create 'permissions' table with foreign keys
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS permissions (
                permission_id INTEGER PRIMARY KEY NOT NULL AUTO_INCREMENT,
                person_id INTEGER NOT NULL,
                door_id INTEGER NOT NULL,
                FOREIGN KEY (person_id) REFERENCES people(person_id) ON DELETE CASCADE,
                FOREIGN KEY (door_id) REFERENCES doors(door_id) ON DELETE CASCADE
            );
        """)

        # Create 'logs' table
        cursor.execute(""" 
            CREATE TABLE IF NOT EXISTS logs (
                log_id INTEGER PRIMARY KEY NOT NULL AUTO_INCREMENT,
                person_id INTEGER NOT NULL,
                door_id INTEGER NOT NULL,
                access_granted BOOLEAN NOT NULL DEFAULT 0,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
            );
        """)
        
        conn.commit()
        print("Database and tables setup completed successfully.")

    # Handle any errors that occur during setup
    except mariadb.Error as err:
        print(f"Error during database setup: {err}", file=sys.stderr)
    # Close the cursor and connection
    finally:
        cursor.close()
        conn.close()

def get_db_connection(database_name="face_rec_db"):
    """
    @brief Establishes a connection to the MariaDB database.
    @param database_name The name of the database to connect to (default: "face_rec_db").
                         Pass None to connect only to the server.
    @return A MariaDB connection object, or None if connection fails.
    """
    try:
        conn = mariadb.connect(
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port'],
            database=database_name
        )
        return conn
    
    except mariadb.Error as err:
        # Handle specific error for unknown database
        if err.errno == 1049 and database_name is not None:
            print(f"Error: Database '{database_name}' does not exist. Attempting to connect without specifying a database...", file=sys.stderr)
            return get_db_connection(database_name=None)

        print(f"Error: Could not connect to MariaDB: {err}", file=sys.stderr)
        return None

def get_all_people():
    """
    @brief Retrieves all records from the 'people' table.
    @return A list of dictionaries, where each dictionary represents a person.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT person_id, name, age, cc_num, contact_num, email, company_name, images_path, encodings_path 
        FROM people ORDER BY person_id ASC
    """)
    
    people_list = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return people_list

def add_person(name, age, cc_num, contact_num, email, company_name):
    """
    @brief Inserts a new person into the database and creates their data folder.
    @details Uses a transaction to ensure database consistency. Creates a folder
    for the person based on their name and newly generated ID.
    @param name Person's name.
    @param age Person's age (can be None).
    @param cc_num Citizen Card number.
    @param contact_num Contact phone number.
    @param email Email address.
    @param company_name Company name (can be None).
    @return The path to the newly created person folder, or False on failure.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        conn.autocommit = False
        placeholder = "PENDING_SETUP"

        query = """
        INSERT INTO people (name, age, cc_num, contact_num, email, company_name, images_path, encodings_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        cursor.execute(query, (name, age, cc_num, contact_num, email, company_name, placeholder, placeholder))

        # Get the ID generated for this new person
        cursor.execute("SELECT LAST_INSERT_ID()")
        new_person_id = cursor.fetchone()[0]

        # Create folder path and update the record
        person_slug = f"{name.strip().lower().replace(' ', '_').replace('.', '_')}_id{new_person_id}"
        person_folder_path = os.path.join(PEOPLE_BASE_DIR, person_slug)

        query_update = "UPDATE people SET images_path = ?, encodings_path = ? WHERE person_id = ?"
        cursor.execute(query_update, (person_folder_path, person_folder_path, new_person_id))

        conn.commit()
        return person_folder_path
    
    except mariadb.Error as err:
        print(f"Error: Failed to add person to Database: {err}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def remove_person(person_id):
    """
    @brief Removes a person from the database and deletes their data folder.
    @param person_id The ID of the person to remove.
    @return "SUCCESS", "NOT_FOUND", or "DB_ERROR" indicating the result.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    person_folder_path = None

    try:
        # Get folder path before deletion
        cursor.execute("SELECT images_path FROM people WHERE person_id = ?", (person_id,))
        result = cursor.fetchone()

        # Check if person exists
        if not result:
            print(f"No person found with ID {person_id}.")
            return "NOT_FOUND"

        # Get the folder path to delete later
        person_folder_path = result["images_path"]

        # Begin transaction
        conn.autocommit = False

        # Delete from dependent tables first, then from 'people'
        cursor.execute("DELETE FROM permissions WHERE person_id = ?", (person_id,))
        cursor.execute("DELETE FROM logs WHERE person_id = ?", (person_id,))
        cursor.execute("DELETE FROM people WHERE person_id = ?", (person_id,))

        # Commit the transaction
        conn.commit()

        # Remove physical folder if it exists
        if person_folder_path and os.path.exists(person_folder_path):
            shutil.rmtree(person_folder_path)

        # Successful deletion
        return "SUCCESS"

    # Handle any database errors
    except mariadb.Error as err:
        print(f"Failed to remove person (rollback executed): {err}", file=sys.stderr)
        conn.rollback()
        return "DB_ERROR"
    finally:
        conn.autocommit = True
        cursor.close()
        conn.close()

def get_person_by_id(person_id):
    """
    @brief Retrieves a single person's data by their ID.
    @param person_id The ID of the person to retrieve.
    @return A dictionary with the person's data, or None if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT * FROM people WHERE person_id = ?", (person_id,))
        person_data = cursor.fetchone()

        if person_data:
            print(f"Person found: {person_data['name']}")
        else:
            print(f"Error: No person found with ID {person_id}.", file=sys.stderr)

        return person_data
    
    except mariadb.Error as err:
        print(f"Failed to retrieve person ID {person_id}: {err}", file=sys.stderr)
        return None
    finally:
        cursor.close()
        conn.close()

def update_person(person_id, age, cc_num, contact_num, email, company_name):
    """
    @brief Updates an existing person's details in the database.
    @param person_id The ID of the person to update.
    @param age New age.
    @param cc_num New Citizen Card number.
    @param contact_num New contact number.
    @param email New email address.
    @param company_name New company name.
    @return True if successful, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        query = """
            UPDATE people
            SET age = ?, cc_num = ?, contact_num = ?, email = ?, company_name = ?
            WHERE person_id = ?
        """
        cursor.execute(query, (age, cc_num, contact_num, email, company_name, person_id))
        conn.commit()
        return True
    
    except mariadb.Error as err:
        print(f"Failed to update person (rollback executed): {err}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def get_all_doors():
    """
    @brief Retrieves all records from the 'doors' table.
    @return A list of dictionaries, where each dictionary represents a door.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute("""SELECT * FROM doors ORDER BY door_id ASC""")
    
    door_list = cursor.fetchall()
    print(f"Retrieved {len(door_list)} records from 'door' table.")
    
    cursor.close()
    conn.close()
    
    return door_list

def add_door(location, description):
    """
    @brief Inserts a new door record into the 'doors' table.
    @param location The location name of the door.
    @param description Optional description of the door.
    @return True if successful, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """INSERT INTO doors (location, description) VALUES (?,?)"""
        cursor.execute(query, (location, description))
        conn.commit()
        return True
    except mariadb.Error as err:
        print(f"Error: Failed to add door to Database: {err}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def remove_door(door_id):
    """
    @brief Removes a door and all associated permissions and logs.
    @param door_id The ID of the door to remove.
    @return "SUCCESS", "NOT FOUND", or "DB_ERROR" indicating the result.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    try:
        cursor.execute("SELECT door_id FROM doors WHERE door_id = ?", (door_id,))
        result = cursor.fetchone()
        
        if not result: 
            return "NOT FOUND"

        conn.autocommit = False

        # Delete dependent records first
        cursor.execute("DELETE FROM permissions WHERE door_id = ?", (door_id,))
        cursor.execute("DELETE FROM logs WHERE door_id = ?", (door_id,))
        cursor.execute("DELETE FROM doors WHERE door_id = ?", (door_id,))
        conn.commit()
        
        print(f"Door with ID {door_id} deleted from database.")
        return "SUCCESS"

    except mariadb.Error as err:
        print(f"Failed to remove door (rollback executed): {err}", file=sys.stderr)
        conn.rollback()
        return "DB_ERROR"
    finally:
        conn.autocommit = True
        cursor.close()
        conn.close()

def get_door_by_id(door_id):
    """
    @brief Retrieves a single door's data by its ID.
    @param door_id The ID of the door to retrieve.
    @return A dictionary with the door's data, or None if not found.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM doors WHERE door_id = ?", (door_id,))
        door_data = cursor.fetchone()

        if door_data:
            print(f"Door found: {door_data['location']}")
        else:
            print(f"No door found with ID {door_id}.")

        return door_data
    
    except mariadb.Error as err:
        print(f"Failed to retrieve door ID {door_id}: {err}", file=sys.stderr)
        return None
    finally:
        cursor.close()
        conn.close()

def update_door(door_id, location, description):
    """
    @brief Updates an existing door's details in the database.
    @param door_id The ID of the door to update.
    @param location New location name.
    @param description New description.
    @return True if successful, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "UPDATE doors SET location = ?, description = ? WHERE door_id = ?"
        cursor.execute(query, (location, description, door_id))
        conn.commit()
        return True
    
    except mariadb.Error as err:
        print(f"Failed to update door (rollback executed): {err}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        cursor.close()
        conn.close()

def get_all_permissions():
    """
    @brief Retrieves all permissions formatted with person names and door locations.
    @return A list of dictionaries containing permission details.
    """
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        query = """
            SELECT 
                perm.permission_id, 
                p.name AS person_name, 
                d.location AS door_location
            FROM permissions perm
            JOIN people p ON perm.person_id = p.person_id
            JOIN doors d ON perm.door_id = d.door_id
            ORDER BY perm.permission_id ASC
        """
        cursor.execute(query)
        permissions_list = cursor.fetchall()
        return permissions_list
        
    except mariadb.Error as err:
        print(f"Failed to retrieve formatted permissions: {err}", file=sys.stderr)
        return []
    finally:
        cursor.close()
        conn.close()

def add_permissions(person_id, door_id_list):
    """
    @brief Adds multiple permissions for a person in a batch.
    @param person_id The ID of the person.
    @param door_id_list A list of door IDs to grant access to.
    @return The number of new permissions successfully added.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cont = 0
    
    try:
        conn.autocommit = False
        
        for door_id in door_id_list:
            # Check if permission already exists to avoid duplicates
            cursor.execute(
                "SELECT 1 FROM permissions WHERE person_id = ? AND door_id = ?",
                (person_id, door_id)
            )
            exists = cursor.fetchone()
            
            if not exists:
                query = "INSERT INTO permissions (person_id, door_id) VALUES (?, ?)"
                cursor.execute(query, (person_id, door_id))
                cont += 1
        
        conn.commit()
        return cont
        
    except mariadb.Error as err:
        print(f"Error: Failed during permissions batch insert (rollback): {err}", file=sys.stderr)
        conn.rollback()
        return 0
    finally:
        conn.autocommit = True
        cursor.close()
        conn.close()

def remove_permission(permission_id):
    """
    @brief Removes a single permission by its ID.
    @param permission_id The ID of the permission to remove.
    @return "SUCCESS", "NOT_FOUND", or "DB_ERROR".
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT 1 FROM permissions WHERE permission_id = ?", (permission_id,))
        result = cursor.fetchone()

        if not result:
            print(f"No permission found with ID {permission_id}.")
            return "NOT_FOUND"

        cursor.execute("DELETE FROM permissions WHERE permission_id = ?", (permission_id,))
        conn.commit()
        
        return "SUCCESS"

    except mariadb.Error as err:
        print(f"Error: Failed to remove permission (rollback executed): {err}", file=sys.stderr)
        conn.rollback()
        return "DB_ERROR"
    finally:
        cursor.close()
        conn.close()

def get_permissions_for_person(person_id):
    """
    @brief Retrieves a list of door IDs that a specific person has access to.
    @param person_id The ID of the person.
    @return A list of door IDs.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        query = "SELECT door_id FROM permissions WHERE person_id = ?"
        cursor.execute(query, (person_id,))
        
        # Flatten the result list of tuples into a simple list of IDs
        door_id_list = [row[0] for row in cursor.fetchall()]
        return door_id_list
        
    except mariadb.Error as err:
        print(f"Error: Failed to retrieve permissions for person {person_id}: {err}", file=sys.stderr)
        return []
    finally:
        cursor.close()
        conn.close()

def set_permissions_for_person(person_id, door_id_list):
    """
    @brief Sets the permissions for a person by replacing all old permissions with a new list.
    @param person_id The ID of the person.
    @param door_id_list The new list of door IDs the person should have access to.
    @return True if successful, False otherwise.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        conn.autocommit = False
        
        # Clear all existing permissions for this person
        cursor.execute("DELETE FROM permissions WHERE person_id = ?", (person_id,))
        
        # Insert new permissions if any are provided
        if door_id_list:
            data_to_insert = [(person_id, int(door_id)) for door_id in door_id_list]
            cursor.executemany("INSERT INTO permissions (person_id, door_id) VALUES (?, ?)", data_to_insert)

        conn.commit()
        return True
        
    except mariadb.Error as err:
        print(f"Error: Failed during permissions set (rollback): {err}", file=sys.stderr)
        conn.rollback()
        return False
    finally:
        conn.autocommit = True
        cursor.close()
        conn.close()

if __name__ == "__main__":
    setup_database()