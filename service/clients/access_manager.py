class AccessManager:
    def __init__(self):
        self.table: dict[tuple[str, str], dict] = {}

    async def grant_access(self, owner_id: str, embedding_id: str, user_id: int):
        key = (owner_id, embedding_id)
        self.table.setdefault(key, {"public": False, "shared_with": set()})
        self.table[key]["shared_with"].add(user_id)

    async def set_public(self, owner_id: str, embedding_id: str, is_public: bool):
        key = (owner_id, embedding_id)
        self.table.setdefault(key, {"public": False, "shared_with": set()})
        self.table[key]["public"] = is_public

    async def has_access(self, current_user_id: int, embedding_id: str) -> bool:
        for (owner, emb_id), access in self.table.items():
            if emb_id == embedding_id:
                return current_user_id == int(owner) or access["public"] or current_user_id in access["shared_with"]
        return False

    async def get_accessible_collections(self, user_id: int):
        result = []
        for (owner, emb_id), access in self.table.items():
            if user_id == int(owner) or access["public"] or user_id in access["shared_with"]:
                result.append(
                    {
                        "embedding_id": emb_id,
                        "owner": int(owner),
                        "public": access["public"],
                    }
                )
        return result

    async def require_access(self, user_id: int, embedding_id: str):
        if not await self.has_access(user_id, embedding_id):
            raise PermissionError(f"User {user_id} has no access to {embedding_id}")


access_manager = AccessManager()
