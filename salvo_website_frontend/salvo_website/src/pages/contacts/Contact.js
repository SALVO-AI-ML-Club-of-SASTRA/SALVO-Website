import React from "react";
import { Link } from "react-router-dom";
import "./Contact.css";
import Nav from "../../components/navigation/Nav";
import Footer from "../../components/footer/Footer";

function Contact() {
    return (
        <div className="contact-page">
            <Nav />
            <h1>Contact Page</h1>
            <p>Welcome to the Contact page!</p>
            <Footer />
        </div>
    );
}

export default Contact;