import React from "react";
import { Link } from "react-router-dom";
import "./Footer.css";

function Footer() {
    return (
        <footer className="footer">
            <div className="footer-content">
                <Link to="/" className="footer-link">Home</Link>
                <Link to="/about" className="footer-link">About</Link>
                <Link to="/contact" className="footer-link">Contact</Link>
            </div>
        </footer>
    );
}

export default Footer;
