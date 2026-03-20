import type { UserInfo, GroupInfo, } from './types';

export const defaultUserInfo: UserInfo = {
    userId: "1",
    userName: "用户",
    avatar: "",
    permission: "SuperUser",
};
export const defaultGroupInfo: GroupInfo = {
    groupId: "1",
    groupName: "会话",
    avatar: "",
};

export let currentUser: UserInfo;
export let userList: UserInfo[] = [];
export function setCurrentUser(userId: string) {
    if (userList.length < 1) {
        userList.push({ ...defaultUserInfo });
        currentUser = userList[0];
    } else {
        currentUser = userList.find(user => user.userId === userId) || userList[0];
    }
    localStorage.setItem("userId", currentUser.userId);
}

export let currentGroup: GroupInfo;
export let groupList: GroupInfo[] = [];
export function setCurrentGroup(groupId: string) {
    if (groupList.length < 1) {
        groupList.push({ ...defaultGroupInfo });
        currentGroup = groupList[0];
    } else {
        currentGroup = groupList.find(group => group.groupId === groupId) || groupList[0];
    }
    localStorage.setItem("groupId", currentGroup.groupId);
}