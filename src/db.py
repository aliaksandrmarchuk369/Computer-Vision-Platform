import mysql.connector

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "user",
    "database": "cv_platform",
}


def ensure_database():
    conn = mysql.connector.connect(
        host=DB_CONFIG["host"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )
    cursor = conn.cursor()
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
    cursor.close()
    conn.close()


def create_table():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS detections")
    cursor.execute(
        """
        CREATE TABLE detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            num_detections INT,
            detections_json JSON,
            conf_threshold FLOAT,
            nms_threshold FLOAT,
            filter_classes VARCHAR(255),
            annotated_image LONGTEXT
        )
    """
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_detection(
    filename: str,
    num_detections: int,
    detections_json: str,
    conf_thr: float,
    nms_thr: float,
    filter_cls: str,
    annotated_image: str,  # base64 string
):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO detections (filename, num_detections, detections_json, conf_threshold, nms_threshold, filter_classes, annotated_image)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """,
        (filename, num_detections, detections_json, conf_thr, nms_thr, filter_cls, annotated_image),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_history_detections():
    """For history page: returns all columns except annotated_image."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT id, filename, timestamp, num_detections, conf_threshold, nms_threshold, filter_classes
        FROM detections
        ORDER BY id DESC
    """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def get_gallery_detections():
    """For gallery page: returns only id, filename, timestamp, annotated_image (non‑empty)."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT filename, annotated_image
        FROM detections
        ORDER BY filename DESC
    """
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def clear_all_detections():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute("TRUNCATE TABLE detections")
    conn.commit()
    cursor.close()
    conn.close()
