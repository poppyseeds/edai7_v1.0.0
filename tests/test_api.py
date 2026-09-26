from io import StringIO

import pandas as pd
from fastapi.testclient import TestClient

from app.api.main import app


def test_plan_accepts_uploaded_csv_with_missing_cells() -> None:
    df = pd.DataFrame(
        {
            "age": list(range(20, 40)),
            "spend": [None if index == 3 else index * 10 for index in range(20)],
            "segment": ["A" if index % 2 else "B" for index in range(20)],
            "target": [index % 2 for index in range(20)],
        }
    )
    csv = StringIO()
    df.to_csv(csv, index=False)

    response = TestClient(app).post(
        "/plan",
        files={"file": ("device_upload.csv", csv.getvalue(), "text/csv")},
        data={"target_column": "target"},
    )

    assert response.status_code == 200, response.text
    assert response.json()["preview"][3]["spend"] is None
