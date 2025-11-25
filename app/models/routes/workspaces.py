from fastapi import APIRouter, HTTPException
from app.database import supabase
from app.models.workspace import WorkspaceCreate, Workspace
from typing import List

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

@router.post("/", response_model=Workspace)
async def create_workspace(workspace: WorkspaceCreate):
    """Create a new workspace"""
    try:
        result = supabase.table("workspaces").insert({
            "name": workspace.name,
            "description": workspace.description
        }).execute()
        
        return result.data[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/", response_model=List[Workspace])
async def list_workspaces():
    """Get all workspaces"""
    try:
        result = supabase.table("workspaces").select("*").execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{workspace_id}", response_model=Workspace)
async def get_workspace(workspace_id: str):
    """Get a specific workspace by ID"""
    try:
        result = supabase.table("workspaces").select("*").eq("id", workspace_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return result.data[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{workspace_id}")
async def delete_workspace(workspace_id: str):
    """Delete a workspace"""
    try:
        result = supabase.table("workspaces").delete().eq("id", workspace_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Workspace not found")
        
        return {"message": "Workspace deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))