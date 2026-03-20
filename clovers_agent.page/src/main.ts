import { userList, setCurrentUser, groupList, setCurrentGroup, currentGroup } from "./code/core";
import { connectCloversServer } from "./code/chat";
import { renderUserList } from "./code/user";
import { sideBarArea } from "./code/sidebar";
import { renderGroupList, showGroupChatHistory } from "./code/sidebar/group";

let showGroupListFlag = false;
function showSideBar(e: PointerEvent) {
    sideBarArea.classList.add('active');
    showGroupListFlag = true;
    e.stopPropagation();
}
function hideSideBar(e: PointerEvent) {
    if (showGroupListFlag && !sideBarArea.contains(e.target as Node)) {
        sideBarArea.classList.remove('active');
        showGroupListFlag = false;
    }
}

const userListBtn = document.getElementById("userListBtn") as HTMLButtonElement;
userListBtn.onclick = () => { renderUserList(); };
const groupListBtn = document.getElementById("groupListBtn") as HTMLButtonElement;
groupListBtn.onclick = showSideBar;

document.addEventListener('click', (e) => {
    hideSideBar(e);
});

document.addEventListener("DOMContentLoaded", () => {
    userList.length = 0;
    try {
        const userListStr = localStorage.getItem("userList");
        if (!userListStr) throw new Error("Default");
        userList.push(...JSON.parse(userListStr));
    } catch (error) {
        console.error("Error parsing userList:", error);
    }
    setCurrentUser(localStorage.getItem("userId") || '1');
    groupList.length = 0;
    try {
        const groupListStr = localStorage.getItem("groupList");
        if (!groupListStr) throw new Error("Default");
        groupList.push(...JSON.parse(groupListStr));
    } catch (error) {
        console.error("Error parsing groupList:", error);
    }
    setCurrentGroup(localStorage.getItem("groupId") || '1');
    renderGroupList();
    showGroupChatHistory(currentGroup.groupId);
    let url = localStorage.getItem("CloversClientConsoleUrl") || "ws://localhost:11000";
    connectCloversServer(url);
});

