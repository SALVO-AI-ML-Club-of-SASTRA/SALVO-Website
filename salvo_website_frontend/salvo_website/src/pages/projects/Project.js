import React from "react";
import { Link } from "react-router-dom";
import "./Project.css";
import Nav from "../../components/navigation/Nav";
import Footer from "../../components/footer/Footer";

function Project() {
    return (
        <div className="project-page">
            <Nav />
            <h1>Project Page</h1>
            <p>Welcome to the Project page!</p>
            <Footer />
        </div>
    );
}

export default Project;