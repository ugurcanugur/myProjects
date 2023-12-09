-- MySQL Workbench Forward Engineering

SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='ONLY_FULL_GROUP_BY,STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

-- -----------------------------------------------------
-- Schema mydb
-- -----------------------------------------------------
-- -----------------------------------------------------
-- Schema hw_database
-- -----------------------------------------------------

-- -----------------------------------------------------
-- Schema hw_database
-- -----------------------------------------------------
CREATE SCHEMA IF NOT EXISTS `hw_database` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci ;
USE `hw_database` ;

-- -----------------------------------------------------
-- Table `hw_database`.`author`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`author` (
  `authorname` VARCHAR(45) NOT NULL,
  `adress` VARCHAR(45) NULL DEFAULT NULL,
  `url` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`authorname`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`publısher`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`publısher` (
  `publisherName` VARCHAR(45) NOT NULL,
  `adress` VARCHAR(45) NULL DEFAULT NULL,
  `phone` VARCHAR(45) NULL DEFAULT NULL,
  `url` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`publisherName`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`book`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`book` (
  `isbn` INT NOT NULL,
  `title` VARCHAR(45) NULL DEFAULT NULL,
  `publishedyear` DATE NULL DEFAULT NULL,
  `price` INT NULL DEFAULT NULL,
  `publishername` VARCHAR(45) NULL,
  `authorname` VARCHAR(45) NOT NULL,
  `authorAdress` VARCHAR(45) NULL,
  PRIMARY KEY (`isbn`, `authorname`),
  INDEX `publishername_idx` (`authorname` ASC) VISIBLE,
  CONSTRAINT `publishername`
    FOREIGN KEY (`authorname`)
    REFERENCES `hw_database`.`publısher` (`publisherName`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `authorname`
    FOREIGN KEY (`authorname`)
    REFERENCES `hw_database`.`author` (`authorname`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`customer`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`customer` (
  `customername` VARCHAR(45) NOT NULL,
  `phone` VARCHAR(45) NULL DEFAULT NULL,
  `adress` VARCHAR(45) NULL DEFAULT NULL,
  `email` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`email`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`customerDetails`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`customerDetails` (
  `age` VARCHAR(45) NULL DEFAULT NULL,
  `gender` BINARY(1) NULL DEFAULT NULL,
  `education` VARCHAR(45) NULL DEFAULT NULL,
  `cıty` VARCHAR(45) NULL DEFAULT NULL,
  `tcno` INT NOT NULL,
  `email` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`tcno`, `email`),
  INDEX `email_idx` (`email` ASC) VISIBLE,
  CONSTRAINT `email`
    FOREIGN KEY (`email`)
    REFERENCES `hw_database`.`customer` (`email`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`warehouse`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`warehouse` (
  `code` INT NOT NULL,
  `phone` VARCHAR(45) NULL DEFAULT NULL,
  `adress` VARCHAR(45) NULL DEFAULT NULL,
  PRIMARY KEY (`code`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`warehouse_book`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`warehouse_book` (
  `count` INT NULL DEFAULT '0',
  `bookısbn` INT NOT NULL,
  `warehousecode` INT NOT NULL,
  INDEX `isbn_idx` (`bookısbn` ASC) VISIBLE,
  INDEX `code_idx` (`warehousecode` ASC) VISIBLE,
  PRIMARY KEY (`warehousecode`, `bookısbn`),
  CONSTRAINT `isbn`
    FOREIGN KEY (`bookısbn`)
    REFERENCES `hw_database`.`book` (`isbn`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `code`
    FOREIGN KEY (`warehousecode`)
    REFERENCES `hw_database`.`warehouse` (`code`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_0900_ai_ci;


-- -----------------------------------------------------
-- Table `hw_database`.`shopingBasket`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`shopingBasket` (
  `id` INT NOT NULL,
  `customerEmail` VARCHAR(45) NOT NULL,
  PRIMARY KEY (`id`, `customerEmail`),
  INDEX `email_idx` (`customerEmail` ASC) VISIBLE,
  CONSTRAINT `email`
    FOREIGN KEY (`customerEmail`)
    REFERENCES `hw_database`.`customer` (`email`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `hw_database`.`shopingBasket_book`
-- -----------------------------------------------------
CREATE TABLE IF NOT EXISTS `hw_database`.`shopingBasket_book` (
  `shopingBasket_id` INT NOT NULL,
  `count` INT NULL,
  `bookISBN` INT NOT NULL,
  PRIMARY KEY (`shopingBasket_id`, `bookISBN`),
  INDEX `ISBN_idx` (`bookISBN` ASC) VISIBLE,
  CONSTRAINT `ISBN`
    FOREIGN KEY (`bookISBN`)
    REFERENCES `hw_database`.`book` (`isbn`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `id`
    FOREIGN KEY (`shopingBasket_id`)
    REFERENCES `hw_database`.`shopingBasket` (`id`)
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
