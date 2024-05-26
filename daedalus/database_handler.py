# ======================================================================================================== #
#                                                                                                          #
#                 FILE:  database.py                                                                       #
#                                                                                                          #
#          DESCRIPTION:                                                                                    #
#                                                                                                          #
#              CREATED:  11:44 AM                                                                          #
#             REVISION:                                                                                    #
#              VERSION:  1.0                                                                               #
#                                                                                                          #
#               AUTHOR:  Sharif Saleki                                                                     #
#                EMAIL: sharif.saleki.gr@dartmouth.edu                                                     #
#          AFFILIATION:  Dartmouth College, Hanover, NH                                                    #
#                                                                                                          #
# ======================================================================================================== #
import sqlite3
from bids import BIDSLayout


def create_connection(db_file):
    """
    Create a connection to the SQLite database specified by db_file.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        print(e)
    return conn


def create_table(conn, create_table_sql):
    """
    Create a table from the create_table_sql statement.
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except sqlite3.Error as e:
        print(e)


def initialize_database(db_path):
    """
    Initialize the database with tables for subjects, sessions, tasks, runs, spaces, events, and confounds.
    """
    sql_create_metadata_table = """
    CREATE TABLE IF NOT EXISTS metadata (
        subject_id TEXT NOT NULL,
        session_id TEXT,
        task TEXT,
        run INTEGER,
        space TEXT,
        file_path TEXT NOT NULL,
        PRIMARY KEY (subject_id, session_id, task, run, file_path)
    );
    """

    sql_create_events_table = """
    CREATE TABLE IF NOT EXISTS events (
        file_path TEXT NOT NULL,
        event_name TEXT,
        onset REAL,
        duration REAL,
        PRIMARY KEY (file_path, event_name, onset)
    );
    """

    sql_create_confounds_table = """
    CREATE TABLE IF NOT EXISTS confounds (
        file_path TEXT NOT NULL,
        confound_name TEXT,
        value REAL,
        PRIMARY KEY (file_path, confound_name)
    );
    """

    # Create a database connection
    conn = create_connection(db_path)

    # Create tables
    if conn is not None:
        create_table(conn, sql_create_metadata_table)
        create_table(conn, sql_create_events_table)
        create_table(conn, sql_create_confounds_table)
    else:
        print("Error! cannot create the database connection.")


def add_bids_metadata(conn, bids_dir):
    """
    Load BIDS metadata from the specified directory and add it to the database.
    """
    layout = BIDSLayout(bids_dir, validate=True)
    c = conn.cursor()

    for file in layout.get():
        entities = file.get_entities()
        subject_id = entities.get('subject', 'N/A')
        session_id = entities.get('session', 'N/A')
        task = entities.get('task', 'N/A')
        run = entities.get('run', None)
        space = entities.get('space', 'N/A')
        file_path = file.path

        c.execute("""
        INSERT INTO metadata (subject_id, session_id, task, run, space, file_path) 
        VALUES (?, ?, ?, ?, ?, ?)
        """, (subject_id, session_id, task, run, space, file_path))

    conn.commit()


def add_event_data(conn, file_path, events):
    """
    Add event data for a specific file.
    """
    c = conn.cursor()
    for event in events:
        c.execute("""
        INSERT INTO events (file_path, event_name, onset, duration)
        VALUES (?, ?, ?, ?)
        """, (file_path, event['event_name'], event['onset'], event['duration']))

    conn.commit()


def add_confound_data(conn, file_path, confounds):
    """
    Add confound data for a specific file.
    """
    c = conn.cursor()
    for confound in confounds:
        c.execute("""
        INSERT INTO confounds (file_path, confound_name, value)
        VALUES (?, ?, ?)
        """, (file_path, confound['confound_name'], confound['value']))

    conn.commit()


def main():
    database = "./fmri_dataset.db"
    bids_directory = "/path/to/your/bids/dataset"

    # Initialize database and tables
    initialize_database(database)

    # Create a database connection
    conn = create_connection(database)

    if conn is not None:
        # Add BIDS metadata
        add_bids_metadata(conn, bids_directory)

        # Example: Add specific event and confound data
        # add_event_data(conn, 'path_to_file', [{'event_name': 'stimulus', 'onset': 10, 'duration': 5}])
        # add_confound_data(conn, 'path_to_file', [{'confound_name': 'motion', 'value': 0.02}])

        # Close the connection
        conn.close()
    else:
        print("Error! cannot create the database connection.")


if __name__ == '__main__':
    main()
