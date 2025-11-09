from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import io
import pandas as pd
from schemas import Dataset, UploadResponse, AnalysisRequest, ChartData, DashboardResponse
from database import db, create_document, get_documents

app = FastAPI(title="DataDash API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/test")
async def test():
    try:
        # Test db list collections
        _ = db.list_collection_names()
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def infer_columns(df: pd.DataFrame) -> List[str]:
    return list(df.columns.astype(str))


@app.post("/upload", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    filename = file.filename
    content = await file.read()

    # Try reading as CSV first; fallback to raw whitespace/sep auto-detect
    try:
        df = pd.read_csv(io.BytesIO(content))
    except Exception:
        try:
            df = pd.read_table(io.BytesIO(content))
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a CSV or tabular text file.")

    cols = infer_columns(df)
    dataset_doc = create_document(
        "dataset",
        {
            "name": filename,
            "columns": cols,
            "row_count": int(len(df)),
        },
    )

    # Persist the raw data rows in a separate collection per dataset id for scale simplicity
    records = df.to_dict(orient="records")
    # Add dataset_id to each record
    for r in records:
        r["dataset_id"] = dataset_doc["id"]
    if records:
        db["datarow"].insert_many(records)

    return UploadResponse(
        dataset_id=dataset_doc["id"],
        name=dataset_doc["name"],
        columns=cols,
        row_count=int(len(df)),
    )


def gen_basic_charts(df: pd.DataFrame) -> List[ChartData]:
    charts: List[ChartData] = []
    # 1. Row count summary
    charts.append(
        ChartData(
            title="Total Rows",
            type="stat",
            labels=["rows"],
            values=[float(len(df))],
            meta={}
        )
    )

    # 2. Column datatype distribution
    dtype_counts = df.dtypes.astype(str).value_counts()
    charts.append(
        ChartData(
            title="Column Types",
            type="bar",
            labels=list(dtype_counts.index.astype(str)),
            values=[float(v) for v in dtype_counts.values],
            meta={}
        )
    )

    # For the next comparisons, pick first numerical and first categorical columns if available
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_categorical_dtype(df[c])]

    # 3. Top categories in first categorical column
    if cat_cols:
        top_cat = df[cat_cols[0]].astype(str).value_counts().head(10)
        charts.append(
            ChartData(
                title=f"Top {cat_cols[0]} Categories",
                type="bar",
                labels=list(top_cat.index.astype(str)),
                values=[float(v) for v in top_cat.values],
                meta={"column": cat_cols[0]}
            )
        )

    # 4. Distribution of first numerical column (binned)
    if num_cols:
        series = df[num_cols[0]].dropna()
        hist = pd.cut(series, bins=10).value_counts().sort_index()
        labels = [f"{i.left:.2f}-{i.right:.2f}" for i in hist.index]
        charts.append(
            ChartData(
                title=f"Distribution of {num_cols[0]}",
                type="bar",
                labels=labels,
                values=[float(v) for v in hist.values],
                meta={"column": num_cols[0]}
            )
        )

    # 5. Correlation heatmap-like pairs (top 5)
    if len(num_cols) >= 2:
        corr = df[num_cols].corr(numeric_only=True).fillna(0)
        pairs = []
        for i, c1 in enumerate(num_cols[:5]):
            for c2 in num_cols[i + 1 : 5]:
                pairs.append((f"{c1} vs {c2}", abs(float(corr.loc[c1, c2]))))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        charts.append(
            ChartData(
                title="Top Numeric Correlations",
                type="bar",
                labels=[p[0] for p in pairs],
                values=[p[1] for p in pairs],
                meta={}
            )
        )

    # 6. Missing values per column
    miss = df.isna().sum()
    charts.append(
        ChartData(
            title="Missing Values per Column",
            type="bar",
            labels=list(miss.index.astype(str)),
            values=[float(v) for v in miss.values],
            meta={}
        )
    )

    # 7. Unique values per column
    uniq = df.nunique(dropna=True)
    charts.append(
        ChartData(
            title="Unique Values per Column",
            type="bar",
            labels=list(uniq.index.astype(str)),
            values=[float(v) for v in uniq.values],
            meta={}
        )
    )

    return charts


@app.get("/datasets")
async def list_datasets(limit: int = 50):
    return get_documents("dataset", {}, limit)


@app.post("/dashboard", response_model=DashboardResponse)
async def dashboard(req: AnalysisRequest):
    # Load rows for dataset
    rows = list(db["datarow"].find({"dataset_id": req.dataset_id}).limit(req.limit))
    if not rows:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("_id", "dataset_id")} for r in rows])

    charts = gen_basic_charts(df)

    # If user specifies x, y, group_by, create an additional focused chart
    if req.x and req.y and req.x in df.columns and req.y in df.columns:
        if pd.api.types.is_numeric_dtype(df[req.y]):
            # grouped mean
            grouped = df.groupby(req.x)[req.y].mean().sort_values(ascending=False).head(20)
            charts.append(
                ChartData(
                    title=f"Average {req.y} by {req.x}",
                    type="bar",
                    labels=list(grouped.index.astype(str)),
                    values=[float(v) for v in grouped.values],
                    meta={"x": req.x, "y": req.y}
                )
            )
        else:
            counts = df.groupby([req.x, req.y]).size().unstack(fill_value=0)
            labels = [str(i) for i in counts.index]
            values = [float(v) for v in counts.sum(axis=1).values]
            charts.append(
                ChartData(
                    title=f"Counts by {req.x} and {req.y}",
                    type="bar",
                    labels=labels,
                    values=values,
                    meta={"x": req.x, "y": req.y}
                )
            )

    return DashboardResponse(charts=charts)
