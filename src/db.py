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
    cursor.execute("DROP TABLE IF EXISTS detection_items")
    cursor.execute("DROP TABLE IF EXISTS detections")
    cursor.execute(
        """
        CREATE TABLE detections (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            filter_classes VARCHAR(255),
            annotated_image LONGTEXT
        )
    """
    )
    cursor.execute(
        """
        CREATE TABLE detection_items (
            id INT AUTO_INCREMENT PRIMARY KEY,
            detection_id INT NOT NULL,
            x1 FLOAT,
            y1 FLOAT,
            x2 FLOAT,
            y2 FLOAT,
            class_id INT,
            class_name VARCHAR(255),
            confidence FLOAT,
            FOREIGN KEY (detection_id) REFERENCES detections(id) ON DELETE CASCADE
        )
    """
    )
    conn.commit()
    cursor.close()
    conn.close()


def insert_detection(
    filename: str,
    filter_cls: str,
    annotated_image: str,
) -> int:
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO detections (filename, filter_classes, annotated_image)
        VALUES (%s, %s, %s)
    """,
        (filename, filter_cls, annotated_image),
    )
    detection_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    conn.close()
    return detection_id


def insert_detection_item(
    detection_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_id: int,
    class_name: str,
    confidence: float,
):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO detection_items (detection_id, x1, y1, x2, y2, class_id, class_name, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
    """,
        (detection_id, x1, y1, x2, y2, class_id, class_name, confidence),
    )
    conn.commit()
    cursor.close()
    conn.close()


def get_history_detections():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        """
        SELECT d.id AS detection_id, d.filename, d.filter_classes,
               di.id AS box_id,
               di.x1, di.y1, di.x2, di.y2, di.class_id, di.class_name, di.confidence
        FROM detections d
        LEFT JOIN detection_items di ON d.id = di.detection_id
        ORDER BY d.id DESC, di.id DESC
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
    cursor.execute("SET FOREIGN_KEY_CHECKS = 0")
    cursor.execute("TRUNCATE TABLE detection_items")
    cursor.execute("TRUNCATE TABLE detections")
    cursor.execute("SET FOREIGN_KEY_CHECKS = 1")
    conn.commit()
    cursor.close()
    conn.close()
