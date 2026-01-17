import os
import io
import threading
import sqlite3
import datetime
import json
from flask import Flask, render_template, request, jsonify, send_file, abort
from model import train_model_background, extract_embedding_for_image, MODEL_PATH

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "attendance.db")
DATASET_DIR = os.path.join(APP_DIR, "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

TRAIN_STATUS_FILE = os.path.join(APP_DIR, "train_status.json")

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------- DB helpers ----------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    roll TEXT,
                    class TEXT,
                    section TEXT,
                    reg_no TEXT,
                    created_at TEXT
                )""")
    c.execute("""CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id INTEGER,
                    name TEXT,
                    timestamp TEXT
                )""")
    conn.commit()
    conn.close()

init_db()

# ---------- Train status helpers ----------
def write_train_status(status_dict):
    with open(TRAIN_STATUS_FILE, "w") as f:
        json.dump(status_dict, f)

def read_train_status():
    if not os.path.exists(TRAIN_STATUS_FILE):
        return {"running": False, "progress": 0, "message": "Not trained"}
    with open(TRAIN_STATUS_FILE, "r") as f:
        return json.load(f)

# ensure initial train status file exists
write_train_status({"running": False, "progress": 0, "message": "No training yet."})

# ---------- Routes ----------
@app.route("/")
def index():
    return render_template("index.html")

# Dashboard comprehensive API for attendance stats
@app.route("/attendance_stats")
def attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    
    # Get all attendance data
    df = pd.read_sql_query("SELECT timestamp FROM attendance", conn)
    
    # Get student statistics
    student_stats = conn.execute("SELECT COUNT(*) as total_students FROM students").fetchone()
    total_students = student_stats[0] if student_stats else 0
    
    # Get attendance statistics
    attendance_stats = conn.execute("""
        SELECT 
            COUNT(DISTINCT student_id) as students_with_attendance,
            COUNT(*) as total_attendance_records,
            MAX(timestamp) as last_attendance
        FROM attendance
    """).fetchone()
    
    conn.close()
    
    if df.empty:
        from datetime import date, timedelta
        days = [(date.today() - datetime.timedelta(days=i)).strftime("%d-%b") for i in range(29, -1, -1)]
        return jsonify({
            "dates": days, 
            "counts": [0]*30,
            "total_students": total_students,
            "students_with_attendance": 0,
            "total_attendance_records": 0,
            "last_attendance": None,
            "attendance_rate": 0
        })
    
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    last_30 = [ (datetime.date.today() - datetime.timedelta(days=i)) for i in range(29, -1, -1) ]
    counts = [ int(df[df['date'] == d].shape[0]) for d in last_30 ]
    dates = [ d.strftime("%d-%b") for d in last_30 ]
    
    # Calculate attendance rate
    students_with_attendance = attendance_stats[0] if attendance_stats else 0
    attendance_rate = (students_with_attendance / total_students * 100) if total_students > 0 else 0
    
    return jsonify({
        "dates": dates, 
        "counts": counts,
        "total_students": total_students,
        "students_with_attendance": students_with_attendance,
        "total_attendance_records": attendance_stats[1] if attendance_stats else 0,
        "last_attendance": attendance_stats[2] if attendance_stats else None,
        "attendance_rate": round(attendance_rate, 1),
        "next_attendance_id": (attendance_stats[1] + 1) if attendance_stats else 1
    })

# Additional API for detailed chart data
@app.route("/detailed_attendance_stats")
def detailed_attendance_stats():
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
    
    # Get attendance by class/section
    class_stats = pd.read_sql_query("""
        SELECT s.class, s.section, COUNT(a.id) as attendance_count
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        GROUP BY s.class, s.section
        ORDER BY s.class, s.section
    """, conn)
    
    # Get daily attendance for last 7 days
    daily_stats = pd.read_sql_query("""
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM attendance
        WHERE DATE(timestamp) >= DATE('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, conn)
    
    # Get top attending students
    top_students = pd.read_sql_query("""
        SELECT s.name, s.roll, s.class, COUNT(a.id) as attendance_count
        FROM students s
        LEFT JOIN attendance a ON s.id = a.student_id
        GROUP BY s.id, s.name, s.roll, s.class
        HAVING COUNT(a.id) > 0
        ORDER BY attendance_count DESC
        LIMIT 10
    """, conn)
    
    conn.close()
    
    return jsonify({
        "class_stats": class_stats.to_dict('records'),
        "daily_stats": daily_stats.to_dict('records'),
        "top_students": top_students.to_dict('records')
    })

# -------- Add student (form) --------
@app.route("/add_student", methods=["GET", "POST"])
def add_student():
    if request.method == "GET":
        return render_template("add_student.html")
    # POST: save student metadata and return student_id
    data = request.form
    name = data.get("name","").strip()
    roll = data.get("roll","").strip()
    cls = data.get("class","").strip()
    sec = data.get("sec","").strip()
    reg_no = data.get("reg_no","").strip()
    if not name:
        return jsonify({"error":"name required"}), 400
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.datetime.utcnow().isoformat()
    c.execute("INSERT INTO students (name, roll, class, section, reg_no, created_at) VALUES (?, ?, ?, ?, ?, ?)",
              (name, roll, cls, sec, reg_no, now))
    sid = c.lastrowid
    conn.commit()
    conn.close()
    # create dataset folder for this student
    os.makedirs(os.path.join(DATASET_DIR, str(sid)), exist_ok=True)
    return jsonify({"student_id": sid})

# -------- Upload face images (after capture) --------
@app.route("/upload_face", methods=["POST"])
def upload_face():
    student_id = request.form.get("student_id")
    if not student_id:
        return jsonify({"error":"student_id required"}), 400
    files = request.files.getlist("images[]")
    saved = 0
    folder = os.path.join(DATASET_DIR, student_id)
    if not os.path.isdir(folder):
        os.makedirs(folder, exist_ok=True)
    for f in files:
        try:
            fname = f"{datetime.datetime.utcnow().timestamp():.6f}_{saved}.jpg"
            path = os.path.join(folder, fname)
            f.save(path)
            saved += 1
        except Exception as e:
            app.logger.error("save error: %s", e)
    return jsonify({"saved": saved})

# -------- Train model (start background thread) --------
@app.route("/train_model", methods=["GET"])
def train_model_route():
    # if already running, respond accordingly
    status = read_train_status()
    if status.get("running"):
        return jsonify({"status":"already_running"}), 202
    # reset status
    write_train_status({"running": True, "progress": 0, "message": "Starting training"})
    # start background thread
    t = threading.Thread(target=train_model_background, args=(DATASET_DIR, lambda p,m: write_train_status({"running": True, "progress": p, "message": m})))
    t.daemon = True
    t.start()
    return jsonify({"status":"started"}), 202

# -------- Train progress (polling) --------
@app.route("/train_status", methods=["GET"])
def train_status():
    return jsonify(read_train_status())

# -------- Mark attendance page --------
@app.route("/mark_attendance", methods=["GET"])
def mark_attendance_page():
    return render_template("mark_attendance.html")

# -------- Recognize face endpoint (POST image) --------
@app.route("/recognize_face", methods=["POST"])
def recognize_face():
    if "image" not in request.files:
        return jsonify({"recognized": False, "error":"no image"}), 400
    img_file = request.files["image"]
    try:
        emb = extract_embedding_for_image(img_file.stream)
        if emb is None:
            return jsonify({"recognized": False, "error":"no face detected"}), 200
        # attempt prediction
        from model import load_model_if_exists, predict_with_model
        clf = load_model_if_exists()
        if clf is None:
            return jsonify({"recognized": False, "error":"model not trained"}), 200
        pred_label, conf = predict_with_model(clf, emb)
        # threshold confidence
        if conf < 0.5:
            return jsonify({"recognized": False, "confidence": float(conf)}), 200
        # find student name
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT name FROM students WHERE id=?", (int(pred_label),))
        row = c.fetchone()
        name = row[0] if row else "Unknown"
        # save attendance record with timestamp
        ts = datetime.datetime.utcnow().isoformat()
        c.execute("INSERT INTO attendance (student_id, name, timestamp) VALUES (?, ?, ?)", (int(pred_label), name, ts))
        attendance_id = c.lastrowid
        conn.commit()
        conn.close()
        return jsonify({"recognized": True, "student_id": int(pred_label), "name": name, "confidence": float(conf), "attendance_id": attendance_id}), 200
    except Exception as e:
        app.logger.exception("recognize error")
        return jsonify({"recognized": False, "error": str(e)}), 500

# -------- Attendance records & filters --------
@app.route("/attendance_record", methods=["GET"])
def attendance_record():
    period = request.args.get("period", "all")  # all, daily, weekly, monthly
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if period == "all":
        # Get all students first
        c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id")
        students = c.fetchall()
        
        # Get attendance records
        c.execute("""SELECT student_id, id, timestamp, 
                     ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY timestamp DESC) as rn
                     FROM attendance""")
        attendance_records = c.fetchall()
        
        # Create attendance lookup
        attendance_lookup = {}
        for record in attendance_records:
            if record[3] == 1:  # Only latest record per student
                attendance_lookup[record[0]] = (record[1], record[2])
        
        # Process each student
        rows = []
        for student in students:
            student_id = student[0]
            attendance_id, timestamp = attendance_lookup.get(student_id, (0, None))
            
            # Check if student has captured photos
            student_folder = os.path.join(DATASET_DIR, str(student_id))
            has_photos = False
            if os.path.exists(student_folder):
                photos = [f for f in os.listdir(student_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                has_photos = len(photos) > 0
            
            # Determine status
            if attendance_id != 0:
                status = 'Present (Attended)'
                display_timestamp = timestamp
            elif has_photos:
                status = 'Present (Photos Captured)'
                display_timestamp = student[6]  # created_at
            else:
                status = 'Absent'
                display_timestamp = None
            
            rows.append((
                attendance_id,
                student_id,
                student[1],  # name
                display_timestamp,
                student[2],  # roll
                student[3],  # class
                student[4],  # section
                student[5],  # reg_no
                student[6],  # created_at
                status
            ))
    else:
        # For specific periods, show only attendance records with student info
        q = """SELECT a.id, a.student_id, a.name, a.timestamp, 
               s.roll, s.class, s.section, s.reg_no, s.created_at,
               'Present' as status
               FROM attendance a 
               LEFT JOIN students s ON a.student_id = s.id"""
        params = ()
        if period == "daily":
            today = datetime.date.today().isoformat()
            q += " WHERE date(a.timestamp) = ?"
            params = (today,)
        elif period == "weekly":
            start = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
            q += " WHERE date(a.timestamp) >= ?"
            params = (start,)
        elif period == "monthly":
            start = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
            q += " WHERE date(a.timestamp) >= ?"
            params = (start,)
        q += " ORDER BY a.timestamp DESC LIMIT 5000"
        c.execute(q, params)
        rows = c.fetchall()
    
    conn.close()
    return render_template("attendance_record.html", records=rows, period=period)

# -------- CSV download --------
@app.route("/download_csv", methods=["GET"])
def download_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get all students with their status based on photo capture OR attendance
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id")
    students = c.fetchall()
    
    # Get attendance records
    c.execute("""SELECT student_id, id, timestamp, 
                 ROW_NUMBER() OVER (PARTITION BY student_id ORDER BY timestamp DESC) as rn
                 FROM attendance""")
    attendance_records = c.fetchall()
    
    # Create attendance lookup
    attendance_lookup = {}
    for record in attendance_records:
        if record[3] == 1:  # Only latest record per student
            attendance_lookup[record[0]] = (record[1], record[2])
    
    # Process each student
    rows = []
    for student in students:
        student_id = student[0]
        attendance_id, timestamp = attendance_lookup.get(student_id, (0, None))
        
        # Check if student has captured photos
        student_folder = os.path.join(DATASET_DIR, str(student_id))
        has_photos = False
        if os.path.exists(student_folder):
            photos = [f for f in os.listdir(student_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            has_photos = len(photos) > 0
        
        # Determine status
        if attendance_id != 0:
            status = 'Present (Attended)'
            display_timestamp = timestamp
        elif has_photos:
            status = 'Present (Photos Captured)'
            display_timestamp = student[6]  # created_at
        else:
            status = 'Absent'
            display_timestamp = None
        
        rows.append((
            attendance_id,
            student_id,
            student[1],  # name
            display_timestamp,
            student[2],  # roll
            student[3],  # class
            student[4],  # section
            student[5],  # reg_no
            student[6],  # created_at
            status
        ))
    conn.close()
    output = io.StringIO()
    output.write("attendance_id,student_id,name,timestamp,roll,class,section,reg_no,student_created_at,status\n")
    for r in rows:
        output.write(f'{r[0]},{r[1]},{r[2]},{r[3] or ""},{r[4] or ""},{r[5] or ""},{r[6] or ""},{r[7] or ""},{r[8] or ""},{r[9]}\n')
    mem = io.BytesIO()
    mem.write(output.getvalue().encode("utf-8"))
    mem.seek(0)
    return send_file(mem, as_attachment=True, download_name="complete_attendance_report.csv", mimetype="text/csv")

# -------- Students API for listing/editing --------
@app.route("/students", methods=["GET"])
def students_list():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, roll, class, section, reg_no, created_at FROM students ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    data = [ {"id":r[0],"name":r[1],"roll":r[2],"class":r[3],"section":r[4],"reg_no":r[5],"created_at":r[6]} for r in rows ]
    return jsonify({"students": data})

# -------- Students with attendance stats --------
@app.route("/students_with_attendance", methods=["GET"])
def students_with_attendance():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get all students with their attendance count
    c.execute("""SELECT s.id, s.name, s.roll, s.class, s.section, s.reg_no, s.created_at,
                 COUNT(a.id) as attendance_count,
                 MAX(a.timestamp) as last_attendance
                 FROM students s 
                 LEFT JOIN attendance a ON s.id = a.student_id
                 GROUP BY s.id, s.name, s.roll, s.class, s.section, s.reg_no, s.created_at
                 ORDER BY s.id DESC""")
    rows = c.fetchall()
    conn.close()
    data = [ {"id":r[0],"name":r[1],"roll":r[2],"class":r[3],"section":r[4],"reg_no":r[5],"created_at":r[6],"attendance_count":r[7],"last_attendance":r[8]} for r in rows ]
    return jsonify({"students": data})

# -------- Students list page --------
@app.route("/students_list", methods=["GET"])
def students_list_page():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Get all students with their attendance count
    c.execute("""SELECT s.id, s.name, s.roll, s.class, s.section, s.reg_no, s.created_at,
                 COUNT(a.id) as attendance_count,
                 MAX(a.timestamp) as last_attendance
                 FROM students s 
                 LEFT JOIN attendance a ON s.id = a.student_id
                 GROUP BY s.id, s.name, s.roll, s.class, s.section, s.reg_no, s.created_at
                 ORDER BY s.id DESC""")
    rows = c.fetchall()
    conn.close()
    return render_template("students_list.html", students=rows)

@app.route("/students/<int:sid>", methods=["DELETE"])
def delete_student(sid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM students WHERE id=?", (sid,))
    c.execute("DELETE FROM attendance WHERE student_id=?", (sid,))
    conn.commit()
    conn.close()
    # also delete dataset folder
    folder = os.path.join(DATASET_DIR, str(sid))
    if os.path.isdir(folder):
        import shutil
        shutil.rmtree(folder, ignore_errors=True)
    return jsonify({"deleted": True})

# ---------------- run ------------------------
if __name__ == "__main__":
    app.run(debug=True)