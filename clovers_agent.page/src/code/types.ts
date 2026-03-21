

export interface UserInfo {
    userId: string;
    userName: string;
    avatar: string;
    permission: "SuperUser" | "Owner" | "Admin" | "Member";
}

export interface GroupInfo {
    groupId: string;
    groupName: string;
    avatar: string;
    flag: boolean;
}