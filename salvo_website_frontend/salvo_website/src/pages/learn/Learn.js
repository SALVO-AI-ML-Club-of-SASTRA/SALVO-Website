import React from "react";
import { Link } from "react-router-dom";
import "./Learn.css";
import Nav from "../../components/navigation/Nav";

function Learn() {
    return (
        <div className="learn-page">
            <Nav />
            <h1>Learn Page</h1>
            <p>Welcome to the Learn page!</p>
        
        </div>
    );
}
export default Learn;